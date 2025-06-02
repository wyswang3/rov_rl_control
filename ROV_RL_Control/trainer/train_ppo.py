#!/usr/bin/env python3
"""
trainer/train_ppo.py

PPO 训练脚本（使用 envs/vector/make_vec_env.py 中的 make_vec_env）：
- 读取 YAML 配置
- 自动选择 CPU/GPU
- 在 rollout 结束后，将 Returns/Advantages 统计记录到 TensorBoard
- 保存最终模型与 VecNormalize 参数
"""

import sys
import yaml
import torch
import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.vector.make_vec_env import make_vec_env


class ROVTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        returns    = buf.returns.flatten()
        advantages = buf.advantages.flatten()
        self.logger.record("rollout/returns_mean", float(returns.mean()))
        self.logger.record("rollout/returns_std",  float(returns.std()))
        self.logger.record("rollout/adv_mean",     float(advantages.mean()))
        self.logger.record("rollout/adv_std",      float(advantages.std()))


def main(config_path: Path):
    # 1) 读取配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2) 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device: {device}")

    # 3) 提取各块配置
    env_params      = cfg.get('env', {})
    reward_params   = cfg.get('reward', {})
    sampling_params = cfg.get('sampling', {})
    ppo_params      = cfg.get('ppo', {})
    train_params    = cfg.get('train', {})

    n_envs     = sampling_params.get('n_envs', 1)
    batch_size = sampling_params.get('batch_size', 512)
    total_timesteps = train_params.get('total_timesteps', 200_000)
    seed       = train_params.get('seed', None)

    # 4) 设随机种子（可选）
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 5) 创建向量化环境
    print(f"[INFO] Creating vectorized env: n_envs = {n_envs}")
    vec_env = make_vec_env(cfg, device, n_envs, False)
    if (seed is not None) and hasattr(vec_env, "seed"):
        vec_env.seed(seed)

    # 6) 准备输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root  = Path('runs') / 'ppo' / timestamp
    ckpt_dir  = run_root / 'checkpoints'
    tb_dir    = run_root / 'tensorboard'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # 7) 处理学习率字段：可以是 float，也可以是 dict（我们只实现线性衰减）
    lr_cfg = ppo_params.get('learning_rate', 3e-4)
    if isinstance(lr_cfg, dict):
        # 期望格式：
        #   learning_rate:
        #     type: "linear"
        #     initial_lr: 1e-4
        lr_type = lr_cfg.get('type', 'linear')
        init_lr = float(lr_cfg.get('initial_lr', 1e-4))
        if lr_type == 'linear':
            # SB3 需要的函数签名： callable(progress_remaining) -> float
            # 其中 progress_remaining 从 1.0 线性降到 0.0
            learning_rate = lambda progress_remaining: init_lr * progress_remaining
        else:
            raise ValueError(f"Unsupported learning_rate type: {lr_type}")
    else:
        # 如果直接写的是数值，就用常数
        learning_rate = float(lr_cfg)

    # 8) 准备 PPO 参数
    gamma         = ppo_params.get('gamma', 0.99)
    gae_lambda    = ppo_params.get('gae_lambda', 0.95)
    ent_coef      = ppo_params.get('ent_coef', 0.0)
    vf_coef       = ppo_params.get('vf_coef', 0.5)
    clip_range    = ppo_params.get('clip_range', 0.2)
    clip_range_vf = ppo_params.get('clip_range_vf', None)
    max_grad_norm = ppo_params.get('max_grad_norm', None)
    n_epochs      = ppo_params.get('n_epochs', 4)

    # 9) 构建 PPO；n_steps = batch_size // n_envs
    n_steps = batch_size // n_envs
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=learning_rate,  # 这里可以是 float 或者 callable
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=str(tb_dir),
        verbose=1,
        device=device,
    )
    print(f"[INFO] Starting training for {total_timesteps} timesteps")

    # 10) 训练并记录日志
    callback = ROVTensorboardCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # 11) 保存最终模型
    final_model_path = ckpt_dir / "ppo_final"
    model.save(str(final_model_path))
    print(f"[INFO] Model saved to {final_model_path}.zip")

    # 12) 如果用了 VecNormalize，就一并保存归一化参数
    try:
        vec_env.save(str(ckpt_dir / "vec_normalize.pkl"))
        print(f"[INFO] VecNormalize stats saved to {ckpt_dir/'vec_normalize.pkl'}")
    except Exception:
        pass

    # 13) 关闭环境
    vec_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO on ROVDynEnv')
    parser.add_argument(
        '-c', '--config',
        type=Path,
        default=Path(__file__).resolve().parent / 'configs' / 'ppo_accel_tuned.yaml',
        help='Path to PPO config file'
    )
    args = parser.parse_args()
    main(args.config)
