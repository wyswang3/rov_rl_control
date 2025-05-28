#!/usr/bin/env python3
"""
scripts/generate_ppo_config.py

生成 PPO 训练配置文件 ppo_accel.yaml（加速度控制任务，CPU 友好版）
"""
import yaml
from pathlib import Path

# —— 默认配置（CPU 上建议并行环境数 ≤ 1，缩短训练时长） —— #
config = {
    'env': {
        'dt':         0.02,   # 时间步长 (s)
        'max_power':  60.0,   # 最大推进器功率 (W)
    },
    'reward': {
        'w_err':   1.0,   # 加速度误差惩罚权重
        'w_jerk':  0.5,   # 加加速度（jerk）惩罚权重
        'w_eng':   0.01,  # 能耗惩罚权重
    },
    'algo':        'PPO',
    'n_envs':      1,          # CPU 上只用 1 个环境
    'total_steps': 200_000,    # 缩短总训练步数
    'lr':          3e-4,
    'gamma':       0.99,
    'gae_lambda':  0.95,
    'batch_size':  512,        # 减小批量大小
    'clip_range':  0.2,
    'ent_coef':    0.0,
    'vf_coef':     0.5,
    # 如果需要，此处可以增加 ppo_epochs、value_lr 等
}

# 写出 YAML 文件到 trainer/configs/ppo_accel.yaml
out_dir = Path(__file__).resolve().parent
out_path = out_dir / 'ppo_accel.yaml'
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, sort_keys=False, allow_unicode=True)

print(f'Generated PPO config for acceleration control: {out_path}')
