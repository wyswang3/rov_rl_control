#!/usr/bin/env python3
"""
trainer/train_ppo.py

优化后的 PPO 训练脚本：
- 读取 YAML 配置
- 支持 CPU / GPU 自动选择
- 在 Windows 下使用 DummyVecEnv，Linux 下可自动切换到 SubprocVecEnv
- 支持环境与算法的并行化封装与归一化
- 保存模型、TensorBoard 日志、并输出关键信息
"""

import os
import sys
import yaml
import torch
import argparse
from datetime import datetime
from pathlib import Path

from envs.rov_dyn_env import ROVDynEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize


def make_env_fn(cfg_env: dict, device: str, seed: int = None):
    """返回一个用于向量化环境的工厂函数。"""

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


def build_vec_env(cfg: dict, device: str):
    """根据配置构建并行环境。Windows 上默认 DummyVecEnv，其他平台可选 SubprocVecEnv。"""
    n_envs = cfg['n_envs']
    # 支持 Windows 和非 Windows 平台
    is_windows = sys.platform.startswith('win')
    EnvClass = DummyVecEnv if is_windows or n_envs == 1 else SubprocVecEnv

    env_fns = [
        make_env_fn(cfg['env'], device, seed=i)
        for i in range(n_envs)
    ]
    vec_env = EnvClass(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    return vec_env


def main(config_path: Path):
    # 加载训练配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device: {device}")

    # 构建并行环境
    vec_env = build_vec_env(cfg, device)
    print(f"[INFO] Built vectorized env with {cfg['n_envs']} environments")

    # 创建输出目录
    checkpoints_dir = Path('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = Path('runs/ppo')
    log_dir.mkdir(parents=True, exist_ok=True)

    # 构建 PPO 模型
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=cfg['lr'],
        n_steps=cfg['batch_size'] // cfg['n_envs'],
        batch_size=cfg['batch_size'],
        n_epochs=cfg.get('ppo_epochs', 4),
        gamma=cfg['gamma'],
        gae_lambda=cfg['gae_lambda'],
        clip_range=cfg['clip_range'],
        ent_coef=cfg['ent_coef'],
        vf_coef=cfg['vf_coef'],
        tensorboard_log=str(log_dir),
        verbose=1,
        device=device
    )
    print(f"[INFO] Starting training for {cfg['total_steps']} timesteps")

    # 训练
    model.learn(total_timesteps=cfg['total_steps'])

    # 保存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    ckpt_path = checkpoints_dir / f"ppo_{timestamp}"
    model.save(str(ckpt_path))
    print(f"[INFO] Model saved to {ckpt_path}")

    # 关闭环境
    vec_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO on ROVDynEnv')
    parser.add_argument(
        '-c', '--config',
        type=Path,
        default=Path(__file__).resolve().parent / 'configs' / 'ppo_hover.yaml',
        help='Path to PPO config file'
    )
    args = parser.parse_args()
    main(args.config)

