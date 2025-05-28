#!/usr/bin/env python3
"""
trainer/train_sac.py

SAC 训练脚本：读取配置、并行环境、训练并保存策略
"""
import os
import yaml
import torch
import argparse
from datetime import datetime
from envs.vector.make_vec_env import make_vec_env
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from models.policy_net import policy_kwargs


def main(config_path: str):
    # Load config
    cfg = yaml.safe_load(open(config_path, 'r'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create vectorized environment
    env_cfg = cfg['env']
    n_envs = cfg['n_envs']
    vec_env = make_vec_env(env_cfg, device, n_envs, asynchronous=True)

    # Wrap with monitor and normalize
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    # Build SAC model
    model = SAC(
        policy='MultiInputPolicy' if False else 'MlpPolicy',
        env=vec_env,
        learning_rate=cfg['lr'],
        buffer_size=cfg.get('buffer_size', 1_000_000),
        batch_size=cfg['batch_size'],
        gamma=cfg['gamma'],
        tau=cfg.get('tau', 0.005),
        ent_coef=cfg['ent_coef'],
        train_freq=cfg.get('train_freq', 1),
        gradient_steps=cfg.get('gradient_steps', 1),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log='runs/sac',
        device=device
    )

    # Start training
    total_steps = cfg['total_steps']
    model.learn(total_timesteps=total_steps)

    # Save policy
    os.makedirs('checkpoints', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = f"checkpoints/sac_{ts}"
    model.save(save_path)
    print(f"SAC model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAC on ROV environment')
    parser.add_argument('-c', '--config', default='trainer/configs/sac_hover.yaml',
                        help='Path to SAC config file')
    args = parser.parse_args()
    main(args.config)
