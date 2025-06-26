# training/evaluate.py
#!/usr/bin/env python3
import os
import argparse
import yaml
import numpy as np
import torch

from env.rov_env import ROVEnv
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.ddpg_agent import DDPGAgent
from training.target_generator import random_pose_target, random_path
from visualization.viz import visualize_episode


def parse_args():
    parser = argparse.ArgumentParser("Evaluate trained RL agent for ROV")
    parser.add_argument("--config", default="training/config.yaml", help="Path to config file")
    parser.add_argument("--algo", choices=["sac","td3","ddpg"], required=True,
                        help="Algorithm name (sac/td3/ddpg)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--episodes", type=int, default=10, help="Total evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode (default from config)")
    parser.add_argument("--output-dir", default="evaluation", help="Directory to save visualizations")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for evaluation")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def select_agent(algo: str, s_dim: int, a_dim: int, cfg: dict):
    mapping = {'sac': SACAgent, 'td3': TD3Agent, 'ddpg': DDPGAgent}
    AgentCls = mapping[algo]
    agent = AgentCls(s_dim, a_dim, 0, cfg.get('hyperparameters', {}), debug=False)
    return agent


def run_episode(env, agent, algo, cfg, seed, max_steps):
    # Initialize episode
    if cfg['task'] == 'pose_control':
        pos_rng = cfg['environment']['position_range']
        tp, tq = random_pose_target(tuple(map(np.array, pos_rng)), True)
        state, _ = env.reset(seed=seed, target_pos=tp, target_quat=tq)
    else:
        wp = cfg['environment']['waypoints']
        bnds = cfg['environment']['bounds']
        path = random_path(wp, tuple(map(np.array, bnds)))
        state, _ = env.reset(seed=seed, path=path)

    final_pos = None
    # step through episode
    for t in range(max_steps):
        # faster binding
        if algo == 'sac':
            action = agent.select_action(state, evaluate=True)
        else:
            action = agent.select_action(state, noise=0.0)
        state, _, done, trunc, _ = env.step(action)
        if done or trunc:
            break
    final_pos = state[:3]
    # determine target for error
    if cfg['task'] == 'pose_control':
        target = env.target_pos
    else:
        idx = min(env._wp_idx, len(env.path)-1)
        target = env.path[idx] if env.path else final_pos
    error = float(np.linalg.norm(final_pos - target))
    return error


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # setup
    task = cfg.get('task', 'pose_control')
    env = ROVEnv(task=task)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    agent = select_agent(args.algo, s_dim, a_dim, cfg)
    agent.load(args.checkpoint)
    agent.set_mode('eval')

    tol = cfg['environment'].get('tolerance', 0.1)
    base_seed = args.seed
    episodes = args.episodes
    # determine max steps
    max_steps = args.max_steps or cfg['hyperparameters'].get('max_steps_per_episode', 1000)

    # Phase 1: quick error scan
    errors = []
    for i in range(episodes):
        errors.append(run_episode(env, agent, args.algo, cfg, base_seed + i, max_steps))
    anomalies = [i for i,e in enumerate(errors) if e > tol]
    print(f"Identified anomalies (error>{tol}): {anomalies}")

    # Phase 2: detailed visualize anomalies only
    for i in anomalies:
        seed = base_seed + i
        # re-simulate with recording
        if task == 'pose_control':
            pos_rng = cfg['environment']['position_range']
            tp, tq = random_pose_target(tuple(map(np.array, pos_rng)), True)
            state, _ = env.reset(seed=seed, target_pos=tp, target_quat=tq)
        else:
            wp = cfg['environment']['waypoints']
            bnds = cfg['environment']['bounds']
            path = random_path(wp, tuple(map(np.array, bnds)))
            state, _ = env.reset(seed=seed, path=path)
        # collect data
        traj = {'positions': [], 'targets': [], 'errors': [], 'actions': []}
        for t in range(max_steps):
            if args.algo == 'sac':
                action = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state, noise=0.0)
            next_state, _, done, trunc, _ = env.step(action)
            pos = next_state[:3]
            if task == 'pose_control':
                target = env.target_pos.copy()
            else:
                idx = min(env._wp_idx, len(env.path)-1)
                target = env.path[idx].copy() if env.path else pos
            err = float(np.linalg.norm(pos - target))
            traj['positions'].append(pos)
            traj['targets'].append(target)
            traj['errors'].append(err)
            traj['actions'].append(action)
            state = next_state
            if done or trunc:
                break
        ep_dir = os.path.join(args.output_dir, f"episode_{i}")
        visualize_episode(traj, ep_dir)
        print(f"Saved visualization for anomaly episode {i} at {ep_dir}")

    print("Evaluation complete.")

if __name__ == '__main__':
    main()
