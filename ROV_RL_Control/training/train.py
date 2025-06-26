# training/train.py
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
from training.target_generator import random_pose_target, generate_trajectory


def parse_args():
    parser = argparse.ArgumentParser("Train RL agent for ROV control")
    parser.add_argument("--config", default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--save-dir", default="training/checkpoints", help="Directory to save checkpoints and logs")
    parser.add_argument("--algo", choices=["sac","td3","ddpg"], default="sac", help="RL algorithm to use")
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


def select_agent(algo: str, state_dim: int, action_dim: int, buffer_size: int, cfg: dict):
    mapping = {'sac': SACAgent, 'td3': TD3Agent, 'ddpg': DDPGAgent}
    AgentCls = mapping.get(algo)
    if AgentCls is None:
        raise ValueError(f"Unsupported algorithm: {algo}")
    return AgentCls(state_dim, action_dim, buffer_size, cfg.get('hyperparameters', {}), debug=False)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f"{args.algo}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    task = cfg.get('task', 'pose_control')
    env = make_env(args.seed, task)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    buffer_size = cfg['hyperparameters'].get('replay_buffer_size', 1_000_000)

    agent = select_agent(args.algo, state_dim, action_dim, buffer_size, cfg)
    agent.set_mode('train')

    logger = Logger(log_dir=os.path.join(save_dir, 'logs'))

    H = cfg['hyperparameters']
    max_eps = H.get('max_episodes', 500)
    max_steps = H.get('max_steps_per_episode', 1000)
    save_int = H.get('save_interval', 50)
    noise = H.get('explore_noise', 0.1)

    train_rewards, train_steps = [], []

    for ep in range(1, max_eps + 1):
        seed_i = args.seed + ep
        # generate task-specific trajectory
        if task == 'pose_control':
            pos_rng = cfg['environment']['position_range']
            tp, tq = random_pose_target(tuple(map(np.array, pos_rng)), True)
            state, _ = env.reset(seed=seed_i, target_pos=tp, target_quat=tq)
        else:
            # continuous trajectory for path_following
            bounds = tuple(map(np.array, cfg['environment']['bounds']))
            pos_traj, quat_traj = generate_trajectory(
                mode='piecewise_linear',
                steps=max_steps,
                num_waypoints=cfg['environment']['waypoints'],
                bounds=bounds,
                seed=seed_i
            )
            state, _ = env.reset(seed=seed_i, path=pos_traj, path_quat=quat_traj)

        ep_reward, steps = 0.0, 0
        stuck = True

        for step in range(max_steps):
            # select action
            if args.algo == 'sac':
                action = agent.select_action(state, evaluate=False)
            else:
                action = agent.select_action(state, noise=noise)

            next_state, reward, done, trunc, _ = env.step(action)
            # check state update
            if not np.allclose(next_state, state, atol=1e-8):
                stuck = False

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            ep_reward += reward
            steps = step + 1
            if done or trunc:
                break

        if stuck:
            raise RuntimeError(f"Episode {ep}: No state change detected, abort training.")

        train_rewards.append(ep_reward)
        train_steps.append(steps)
        logger.log_scalar('Train/EpisodeReward', ep_reward, ep)
        print(f"Episode {ep}/{max_eps} | Reward: {ep_reward:.2f} | Steps: {steps}")

        if ep % save_int == 0:
            ckpt = os.path.join(save_dir, f"{args.algo}_ep{ep}.pth")
            agent.save(ckpt)

    final_ckpt = os.path.join(save_dir, f"{args.algo}_final.pth")
    agent.save(final_ckpt)
    env.close()
    logger.close()

if __name__ == '__main__':
    main()
