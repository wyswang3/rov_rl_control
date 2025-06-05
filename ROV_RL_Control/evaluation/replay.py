# 文件：rov_rl_control/training/replay.py

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


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a trained RL agent and record video")
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
        default=5,
        help="Number of episodes to replay"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="replay_videos",
        help="Directory to save recorded videos"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(seed: int, video_path: str):
    """
    创建 ROV 环境并包装为 RecordVideo wrapper，录制视频到指定路径。
    """
    base_env = ROVEnv(render_mode=None, disturbance_enable=False)
    base_env.action_space.seed(seed)
    base_env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 使用 Gymnasium 的 RecordVideo wrapper 录制视频
    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=video_path,
        episode_trigger=lambda ep_id: True,  # 每个 episode 都录制
        name_prefix="rov_replay"
    )
    # 取消自动关闭环境时清理视频文件，以免只生成首个
    env.unwrapped.enable_auto_render = False
    return env


def select_agent(algo: str, state_dim: int, action_dim: int, config: dict):
    """
    根据算法名称创建对应的 Agent，并加载其网络架构与超参数。
    """
    replay_size = config["hyperparameters"]["replay_buffer_size"]
    if algo == "sac":
        return SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=replay_size,
            config=config["hyperparameters"]
        )
    elif algo == "td3":
        return TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=replay_size,
            config=config["hyperparameters"]
        )
    elif algo == "ddpg":
        return DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            replay_buffer_size=replay_size,
            config=config["hyperparameters"]
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


def replay_and_record(env: gym.Env, agent, algo: str, episodes: int):
    """
    在环境中使用已加载的 agent，运行若干 episodes 并录制视频。
    """
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            if algo in ["td3", "ddpg"]:
                action = agent.select_action(state, noise=0.0)
            else:  # sac
                action = agent.select_action(state, evaluate=True)

            next_state, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            state = next_state

        print(f"[Replay] Episode {ep} Reward: {ep_reward:.2f}")

    # 等待 wrapper 写完视频
    env.close()


def main():
    args = parse_args()
    config = load_config(args.config)

    # 固定随机种子
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 准备输出目录：按时间戳创建子目录
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    video_save_dir = os.path.join(args.output_dir, f"{args.algo}_replay_{timestamp}")
    os.makedirs(video_save_dir, exist_ok=True)

    # 创建环境并包装录像
    env = make_env(seed, video_save_dir)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 创建 Agent 并加载模型
    agent = select_agent(args.algo, state_dim, action_dim, config)
    agent.load(args.model_path)
    print(f"Loaded {args.algo.upper()} model from {args.model_path}")

    # 运行 replay 并录制视频
    replay_and_record(env, agent, args.algo, args.episodes)

    print(f"Replay videos saved under: {video_save_dir}")


if __name__ == "__main__":
    main()
