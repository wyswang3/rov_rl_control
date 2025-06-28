#!/usr/bin/env python3
import os
import argparse
import yaml
import time
import numpy as np
import torch

from env.rov_env import ROVEnv
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.ddpg_agent import DDPGAgent
from training.logger import Logger
from training.target_generator import random_pose_target, random_path


def parse_args():
    parser = argparse.ArgumentParser("Train RL agent for ROV control")
    parser.add_argument(
        "--config", type=str, default="training/config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--algo", type=str, choices=["sac", "td3", "ddpg"],
        default="sac", help="Algorithm to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--save-dir", type=str, default="training/checkpoints",
        help="Directory to save checkpoints and logs"
    )
    parser.add_argument(
        "--record-interval", type=int, default=100,
        help="Record trajectories every N episodes"
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def make_env(seed: int, task: str) -> ROVEnv:
    env = ROVEnv(task=task)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def select_agent(algo: str, state_dim: int, action_dim: int, buffer_size: int, hyper: dict):
    mapping = {
        'sac': SACAgent,
        'td3': TD3Agent,
        'ddpg': DDPGAgent
    }
    Agent = mapping.get(algo)
    if Agent is None:
        raise ValueError(f"Unsupported algorithm: {algo}")
    return Agent(state_dim, action_dim, buffer_size, hyper)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # set global seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare save directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f"{args.algo}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # initialize environment and agent
    task = cfg.get('task', 'pose_control')
    env = make_env(args.seed, task)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    buffer_size = cfg['hyperparameters'].get('replay_buffer_size', 1_000_000)
    agent = select_agent(
        args.algo, state_dim, action_dim,
        buffer_size, cfg['hyperparameters']
    )

    # logger
    logger = Logger(log_dir=os.path.join(save_dir, 'logs'))

    # hyperparameters
    H = cfg['hyperparameters']
    max_eps = H.get('max_episodes', 500)
    max_steps = H.get('max_steps_per_episode', 1000)
    save_int = H.get('save_interval', 50)
    explore_noise = H.get('explore_noise', 0.1)
    record_int = args.record_interval

    # storage for metrics and trajectories
    train_rewards = []
    train_steps = []
    recorded = {}

    for ep in range(1, max_eps + 1):
        seed_i = args.seed + ep
        # reset environment with target
        if task == 'pose_control':
            pos_rng = cfg['environment']['position_range']
            tp, tq = random_pose_target(tuple(map(np.array, pos_rng)), True)
            state, _ = env.reset(
                seed=seed_i,
                target_pos=tp,
                target_quat=tq
            )
            target = {'pos': tp, 'quat': tq}
        else:
            wp = cfg['environment']['waypoints']
            bnds = tuple(map(np.array, cfg['environment']['bounds']))
            path = random_path(wp, bnds)
            state, _ = env.reset(seed=seed_i, path=path)
            target = {'path': path}

        ep_reward = 0.0
        steps = 0
        # prepare trajectory recording
        if ep % record_int == 0:
            traj_pos = []
            traj_quat = []

        for step in range(1, max_steps + 1):
            # select action
            if args.algo == 'sac':
                action = agent.select_action(state, evaluate=False)
            else:
                action = agent.select_action(state, noise=explore_noise)

            next_state, reward, done, trunc, info = env.step(action)
            ep_reward += reward
            steps = step

            # record raw state
            if ep % record_int == 0:
                raw = env.state.copy()
                traj_pos.append(raw[0:3])
                traj_quat.append(raw[3:7])

            # store and update
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if done or trunc:
                break

        # log metrics
        train_rewards.append(ep_reward)
        train_steps.append(steps)
        logger.log_scalar('Train/EpisodeReward', ep_reward, ep)
        print(f"Episode {ep}/{max_eps} | Reward: {ep_reward:.2f} | Steps: {steps}")

        # periodic saving
        if ep % save_int == 0:
            ckpt = os.path.join(save_dir, f"{args.algo}_ep{ep}.pth")
            agent.save(ckpt)
        # record trajectories
        if ep % record_int == 0:
            recorded[f"ep_{ep}"] = {
                'positions': np.array(traj_pos, dtype=np.float32),
                'quaternions': np.array(traj_quat, dtype=np.float32),
                'target': target
            }

    # final save
    final_path = os.path.join(save_dir, f"{args.algo}_final.pth")
    agent.save(final_path)
    # dump recorded trajectories
    if recorded:
        np.savez(
            os.path.join(save_dir, 'recorded_trajectories.npz'),
            **recorded
        )

    env.close()
    logger.close()


if __name__ == '__main__':
    main()
