#!/usr/bin/env python3
"""
生成 PPO 训练配置文件 ppo_hover.yaml
"""
import yaml
from pathlib import Path

# 定义默认配置
config = {
    'env': {
        'dt': 0.02,           # 时间步长 (s)
        'max_power': 480.0,   # 最大推进器功率 (W)
    },
    'reward': {
        'pos': 1.0,           # 位置误差惩罚权重
        'ori': 0.3,           # 姿态误差惩罚权重
        'energy': 0.01,       # 能耗惩罚权重
    },
    'algo': 'PPO',
    'total_steps': 5_000_000,
    'n_envs': 8,
    'lr': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'batch_size': 8192,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
}

# 写出 YAML 文件
out_path = Path(__file__).resolve().parent / 'ppo_hover.yaml'
with open(out_path, 'w') as f:
    yaml.dump(config, f, sort_keys=False)
print(f'Generated PPO config: {out_path}')
