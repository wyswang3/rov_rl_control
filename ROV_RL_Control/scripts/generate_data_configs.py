#!/usr/bin/env python3
"""
scripts/generate_data_configs.py

生成静态数据：
  1. 域随机化配置 data/domain_rand.yaml
  2. 悬停轨迹文件 data/ref_trajs/hover_traj.npy
"""
import os
import yaml
import numpy as np
from pathlib import Path

# ----- 配置参数 -----
dt = 0.02            # 时间步长 (s)
T  = 20.0            # 总时长 (s)
steps = int(T / dt)

# 域随机化范围
domain_rand = {
    'mass_scale':      [0.9, 1.1],     # 质量缩放因子
    'drag_scale':      [0.8, 1.2],     # 阻尼缩放因子
    'accel_noise_std': 0.02,           # 线加速度噪声 (m/s^2)
    'gyro_noise_std':  0.01,           # 角加速度噪声 (rad/s)
}

# 悬停轨迹：所有参考状态置零 (steps × 6)
hover_traj = np.zeros((steps, 6), dtype=np.float32)

# ----- 写出文件 -----
# 根目录 = 项目根/data
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
REF_DIR  = DATA_DIR / 'ref_trajs'

# 创建目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

# 1) 域随机化 YAML
domain_file = DATA_DIR / 'domain_rand.yaml'
with open(domain_file, 'w') as f:
    yaml.dump(domain_rand, f, sort_keys=False)
print(f'Generated domain randomization config: {domain_file}')

# 2) 悬停轨迹 NumPy
hover_file = REF_DIR / 'hover_traj.npy'
np.save(hover_file, hover_traj)
print(f'Generated hover trajectory file: {hover_file}  (shape={hover_traj.shape})')
