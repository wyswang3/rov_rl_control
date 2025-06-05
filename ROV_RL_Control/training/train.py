# 文件：rov_rl_control/training/train.py

import os
import argparse
import yaml
import time

import torch
import gymnasium as gym
import numpy as np

from env.rov_env import ROVEnv
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.ddpg_agent import DDPGAgent
from agents.replay_buffer import ReplayBuffer

from training.logger import Logger  # 假设 logger.py 提供了一个 Logger 类，封装了 TensorBoardWriter 等功能


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent for ROV control")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save models and logs"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["sac", "td3", "ddpg"],
        default="sac",
        help="Which algorithm to train (sac / td3 / ddpg)"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(seed: int):
    env = ROVEnv(render_mode=None, disturbance_enable=True)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return env


def select_agent(algo: str, state_dim: int, action_dim: int, replay_size: int, config: dict):
    """
    根据 algo 字符串选择并创建对应的 Agent 实例。
    """
    if algo == "sac":
        return SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=replay_size,
            config=config["sac"]
        )
    elif algo == "td3":
        return TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=replay_size,
            config=config["td3"]
        )
    elif algo == "ddpg":
        return DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=replay_size,
            config=config["ddpg"]
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


def main():
    args = parse_args()
    config = load_config(os.path.join("rov_rl_control", "training", args.config))

    # Set random seeds
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 创建保存目录
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path = os.path.join(args.save_dir, f"{args.algo}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    # 初始化环境
    env = make_env(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 初始化 Agent
    replay_size = config.get("replay_buffer_size", 1_000_000)
    agent = select_agent(
        algo=args.algo,
        state_dim=state_dim,
        action_dim=action_dim,
        replay_size=replay_size,
        config=config["hyperparameters"]
    )

    # 初始化日志
    logger = Logger(log_dir=os.path.join(save_path, "logs"))

    # 训练参数
    max_episodes = config["hyperparameters"].get("max_episodes", 500)
    max_steps = config["hyperparameters"].get("max_steps_per_episode", 1000)
    eval_interval = config["hyperparameters"].get("eval_interval", 20)
    save_interval = config["hyperparameters"].get("save_interval", 50)

    total_steps = 0

    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(1, max_steps + 1):
            # 1. 选择动作（训练时带探索噪声）
            if args.algo in ["td3", "ddpg"]:
                action = agent.select_action(state, noise=config["hyperparameters"].get("explore_noise", 0.1))
            else:  # sac
                action = agent.select_action(state, evaluate=False)

            # 2. 执行动作
            next_state, reward, done, truncated, info = env.step(action)

            # 3. 存储样本
            agent.store_transition(state, action, reward, next_state, done or truncated)

            # 4. 更新网络
            agent.update()

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done or truncated:
                break

        # 记录该 Episode 的回报
        logger.log_scalar("Train/EpisodeReward", episode_reward, ep)
        print(f"Episode {ep} | Reward: {episode_reward:.2f} | Steps: {step}")

        # 定期保存模型
        if ep % save_interval == 0:
            model_file = os.path.join(save_path, f"{args.algo}_episode_{ep}.pth")
            agent.save(model_file)
            print(f"Saved model at Episode {ep} to {model_file}")

        # 定期评估
        if ep % eval_interval == 0:
            avg_eval_reward = evaluate_policy(env, agent, args.algo, config["hyperparameters"], seed)
            logger.log_scalar("Eval/AverageReward", avg_eval_reward, ep)
            print(f"Evaluation after Episode {ep}: Avg Reward = {avg_eval_reward:.2f}")

    # 最后保存一次
    final_model = os.path.join(save_path, f"{args.algo}_final.pth")
    agent.save(final_model)
    print(f"Training completed. Final model saved to {final_model}")

    env.close()
    logger.close()


def evaluate_policy(env: gym.Env, agent, algo: str, hyperparams: dict, seed: int, eval_episodes: int = 5) -> float:
    """
    评估当前策略在多个 Episode 上的平均回报（不带噪声）。
    """
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            if algo in ["td3", "ddpg"]:
                action = agent.select_action(state, noise=0.0)  # 评估时不加噪声
            else:  # sac
                action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, info = env.step(action)
            ep_reward += reward
        avg_reward += ep_reward
    return avg_reward / eval_episodes


if __name__ == "__main__":
    main()
