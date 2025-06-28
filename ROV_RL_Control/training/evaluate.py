#!/usr/bin/env python3
import os
import argparse
import yaml
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from env.rov_env import ROVEnv
from agents.sac_agent import SACAgent
from agents.td3_agent import TD3Agent
from agents.ddpg_agent import DDPGAgent
from training.target_generator import random_pose_target, generate_trajectory

def parse_args():
    parser = argparse.ArgumentParser("Evaluate trained RL agent for ROV control and generate reports")
    parser.add_argument(
        "--config", type=str, default="training/config.yaml",
        help="Path to training config.yaml"
    )
    parser.add_argument(
        "--algo", choices=["sac","td3","ddpg"], required=True,
        help="Algorithm of the model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to agent checkpoint (.pth)"
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Max steps per episode (override config)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation",
        help="Directory to save evaluation outputs"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed"
    )
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def make_env(task, seed):
    env = ROVEnv(task=task)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return env

def select_agent(algo, state_dim, action_dim, hyper):
    mapping = {'sac': SACAgent, 'td3': TD3Agent, 'ddpg': DDPGAgent}
    AgentCls = mapping[algo]
    return AgentCls(state_dim, action_dim, hyper.get('replay_buffer_size', 1_000_000), hyper)

def run_episode(env, agent, algo, cfg, seed, max_steps):
    task = cfg.get('task', 'pose_control')
    # prepare target trajectory or pose
    if task == 'pose_control':
        pos_rng = tuple(map(np.array, cfg['environment']['position_range']))
        tp, tq = random_pose_target(pos_rng, True)
        state, _ = env.reset(seed=seed, target_pos=tp, target_quat=tq)
        ref_traj = np.tile(tp[None,:], (max_steps,1))
    else:
        env_hp = cfg['environment']
        mode = env_hp.get('trajectory_mode', 'piecewise_linear')
        pos_traj, quat_traj = generate_trajectory(
            mode=mode,
            steps=max_steps,
            num_waypoints=env_hp['waypoints'],
            bounds=tuple(map(np.array, env_hp['bounds'])),
            seed=seed
        )
        state, _ = env.reset(seed=seed, path=pos_traj, path_quat=quat_traj)
        ref_traj = pos_traj

    traj_pos, traj_quat = [], []
    total_reward = 0.0
    for step in range(max_steps):
        if algo == 'sac':
            action = agent.select_action(state, evaluate=True)
        else:
            action = agent.select_action(state, noise=0.0)
        next_state, r, done, trunc, info = env.step(action)
        total_reward += r
        traj_pos.append(env.state[:3].copy())
        traj_quat.append(env.state[3:7].copy())
        state = next_state
        if done or trunc:
            break
    return np.array(traj_pos), np.array(traj_quat), ref_traj[:len(traj_pos)], total_reward

def main():
    args = parse_args()
    cfg = load_config(args.config)
    hp = cfg['hyperparameters']
    max_steps = args.max_steps or hp.get('max_steps_per_episode', 1000)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output_dir, f"{args.algo}_eval_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    env = make_env(cfg.get('task','pose_control'), args.seed)
    sd = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    agent = select_agent(args.algo, sd, ad, hp)
    agent.load(args.checkpoint)
    print(f"Loaded {args.algo.upper()} checkpoint from {args.checkpoint}")

    all_errors, all_trajs, all_refs, all_rewards = [], [], [], []

    for ep in range(1, args.episodes+1):
        traj_p, traj_q, ref_p, tot_r = run_episode(env, agent, args.algo, cfg, args.seed+ep, max_steps)
        err = np.linalg.norm(traj_p[-1] - ref_p[-1])
        all_errors.append(err)
        all_trajs.append(traj_p)
        all_refs.append(ref_p)
        all_rewards.append(tot_r)
        print(f"Episode {ep}/{args.episodes} | Return {tot_r:.2f} | End-error {err:.4f}")

    np.savez(
        os.path.join(out_dir,'eval_data.npz'),
        trajectories=np.array(all_trajs, dtype=object),
        references=np.array(all_refs, dtype=object),
        rewards=np.array(all_rewards),
        errors=np.array(all_errors)
    )

    plt.figure()
    plt.plot(range(1,len(all_errors)+1), all_errors, marker='o', label='Final error')
    plt.title('Final Position Error per Episode', fontname='Times New Roman')
    plt.xlabel('Episode', fontname='Times New Roman')
    plt.ylabel('Error [m]', fontname='Times New Roman')
    plt.legend(prop={'family':'Times New Roman'})
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'error_curve.png'), dpi=200)
    plt.close()

    last_p = all_trajs[-1]
    last_ref = all_refs[-1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(last_p[:,0], last_p[:,1], last_p[:,2], label='Actual')
    ax.plot(last_ref[:,0], last_ref[:,1], last_ref[:,2], '--', label='Target')
    ax.set_title('Trajectory Comparison', fontname='Times New Roman')
    ax.legend(prop={'family':'Times New Roman'})
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,'trajectory_comparison.png'), dpi=200)
    plt.close()

    env.close()

if __name__ == '__main__':
    main()
