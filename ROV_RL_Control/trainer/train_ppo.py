#!/usr/bin/env python3
"""
trainer/train_ppo.py

PPO 训练脚本：读取配置、并行环境、训练并保存策略
"""
import os
import yaml
import torch
import argparse
from datetime import datetime
from ..envs.rov_dyn_env import ROVDynEnv
from gymnasium.vector import AsyncVectorEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize


def make_env_fn(cfg_env: dict, device: str, seed: int = None):
    """返回一个可用于 AsyncVectorEnv 的环境初始化函数"""
    def _init():
        env = ROVDynEnv(
            dt=cfg_env['dt'],
            max_power=cfg_env['max_power'],
            device=device
        )
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def main(config_path: str):
    # 加载训练配置
    cfg = yaml.safe_load(open(config_path, 'r'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建并行环境
    n_envs = cfg['n_envs']
    env_fns = [make_env_fn(cfg['env'], device, seed=i) for i in range(n_envs)]
    vec_env = AsyncVectorEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    # 构建 PPO 模型
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=cfg['lr'],
        n_steps=cfg['batch_size'] // n_envs,
        batch_size=cfg['batch_size'],
        gamma=cfg['gamma'],
        gae_lambda=cfg['gae_lambda'],
        clip_range=cfg['clip_range'],
        ent_coef=cfg['ent_coef'],
        vf_coef=cfg['vf_coef'],
        verbose=1,
        tensorboard_log='runs/ppo',
        device=device
    )

    # 开始训练
    total_steps = cfg['total_steps']
    model.learn(total_timesteps=total_steps)

    # 保存策略
    os.makedirs('checkpoints', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = f"checkpoints/ppo_{ts}"
    model.save(save_path)
    print(f"PPO model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO on ROV environment')
    parser.add_argument('-c', '--config', default='trainer/configs/ppo_hover.yaml',
                        help='Path to PPO config file')
    args = parser.parse_args()
    main(args.config)
