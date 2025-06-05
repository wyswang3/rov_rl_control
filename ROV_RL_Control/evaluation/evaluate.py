# 文件：rov_rl_control/training/evaluate.py

import os
import argparse
import yaml
import time

import torch
import numpy as np
import gymnasium as gym

from env.rov_env import ROVEnv
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.ddpg_agent import DDPGAgent
from training.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent on the ROV environment")
    parser.add_argument(
        "--config",
        type=str,
        default="rov_rl_control/training/config.yaml",
        help="Path to the YAML configuration file used during training"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["sac", "td3", "ddpg"],
        required=True,
        help="Algorithm type of the saved model (sac / td3 / ddpg)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model (.pth) to load"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether to render the environment during evaluation"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="eval_logs",
        help="Directory to save evaluation logs (TensorBoard)"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(seed: int, render: bool = False):
    """
    创建并返回一个 ROV 环境实例，固定随机种子。
    """
    env = ROVEnv(render_mode="human" if render else None, disturbance_enable=False)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return env


def select_agent(algo: str, state_dim: int, action_dim: int, config: dict):
    """
    根据算法名称创建对应的 Agent，并加载其网络架构与超参数。
    """
    if algo == "sac":
        return SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=config["hyperparameters"]["replay_buffer_size"],
            config=config["hyperparameters"]
        )
    elif algo == "td3":
        return TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=config["hyperparameters"]["replay_buffer_size"],
            config=config["hyperparameters"]
        )
    elif algo == "ddpg":
        return DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=config["hyperparameters"]["replay_buffer_size"],
            config=config["hyperparameters"]
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


def evaluate_policy(env: gym.Env, agent, algo: str, episodes: int, render: bool, logger: Logger):
    """
    在给定环境和 agent 上运行指定数量的评估 Episodes，
    记录每集回报并输出平均值。

    Args:
        env: 已初始化的 ROVEnv 实例
        agent: 已加载模型参数的 Agent
        algo: "sac" / "td3" / "ddpg"
        episodes: 评估集数
        render: 是否在每步渲染环境（仅在需要可视化时开启）
        logger: Logger 实例，用于记录评估回报到 TensorBoard
    """
    total_reward = 0.0
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            if algo in ["td3", "ddpg"]:
                action = agent.select_action(state, noise=0.0)
            else:  # sac
                action = agent.select_action(state, evaluate=True)

            next_state, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            if render:
                env.render()

            state = next_state

        total_reward += ep_reward
        logger.log_scalar("Eval/EpisodeReward", ep_reward, ep)
        print(f"[Eval] Episode {ep} Reward: {ep_reward:.2f}")

    avg_reward = total_reward / episodes
    logger.log_scalar("Eval/AverageReward", avg_reward, 0)
    print(f"[Eval] Average Reward over {episodes} episodes: {avg_reward:.2f}")
    return avg_reward


def main():
    args = parse_args()
    config = load_config(args.config)

    # Set seeds
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 准备保存日志目录
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = f"{args.algo}_eval_{timestamp}"
    log_path = os.path.join(args.log_dir, run_name)
    os.makedirs(log_path, exist_ok=True)
    logger = Logger(log_dir=args.log_dir, run_name=run_name)

    # 创建环境
    env = make_env(seed, render=args.render)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 创建 Agent 并加载模型
    agent = select_agent(args.algo, state_dim, action_dim, config)
    agent.load(args.model_path)
    print(f"Loaded {args.algo.upper()} model from {args.model_path}")

    # 评估
    avg_reward = evaluate_policy(env, agent, args.algo, args.episodes, args.render, logger)

    print(f"Final Average Reward: {avg_reward:.2f}")

    # 记录最终评估结果到日志
    logger.log_text("Eval/Config", yaml.dump(config), 0)
    logger.log_scalar("Eval/AvgReward_Final", avg_reward, 0)

    logger.close()
    env.close()


if __name__ == "__main__":
    main()
