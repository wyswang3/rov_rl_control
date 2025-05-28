#!/usr/bin/env python3
"""
scripts/generate_data_configs.py

生成静态数据：
  1. 域随机化配置 data/domain_rand.yaml
  2. 参考轨迹文件 data/ref_trajs/*.npy
     - hover_traj.npy
     - circle_traj.npy
     - z_wave_traj.npy
     - accel_ref_traj.npy      ← 新增：连续加速度参考
"""
import os
import yaml
import numpy as np
from pathlib import Path

# ----- 基本参数 -----
dt    = 0.02    # 时间步长 (s)
T     = 20.0    # 总时长 (s)
steps = int(T / dt)
t     = np.linspace(0, T, steps, endpoint=False)

# ----- 域随机化范围 -----
domain_rand = {
    'mass_scale':      [0.9, 1.1],
    'drag_scale':      [0.8, 1.2],
    'accel_noise_std': 0.02,
    'gyro_noise_std':  0.01,
}

# ----- 轨迹定义 -----
# 1) 悬停轨迹
hover_traj = np.zeros((steps, 6), dtype=np.float32)

# 2) 圆形轨迹（位置/姿态）
radius = 1.0
omega  = 2 * np.pi / T
x_circ = radius * np.cos(omega * t)
y_circ = radius * np.sin(omega * t)
z_circ = np.zeros_like(t)
yaw_c   = omega * t
circle_traj = np.stack([
    x_circ, y_circ, z_circ,
    np.zeros_like(t), np.zeros_like(t), yaw_c
], axis=1).astype(np.float32)

# 3) 竖直摆动轨迹（位置/姿态）
amp_z = 0.5
z_wave = amp_z * np.sin(2 * np.pi * t / T)
z_wave_traj = np.stack([
    np.zeros_like(t), np.zeros_like(t), z_wave,
    np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
], axis=1).astype(np.float32)

# 4) 加速度参考轨迹（线加速度 + 角加速度）
#    - 线加速度在 X 方向做 0.1 m/s^2 幅值正弦
#    - 角加速度全零
amp_a = 0.1  # m/s^2
acc_x = amp_a * np.sin(2 * np.pi * t / T)
accel_ref_traj = np.zeros((steps, 6), dtype=np.float32)
accel_ref_traj[:, 0] = acc_x
# accel_ref_traj[:, 1:3] = 0 (默认)
# accel_ref_traj[:, 3:6] = 0 (角加速度)

# ----- 写出文件 -----
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
REF_DIR  = DATA_DIR / 'ref_trajs'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REF_DIR,  exist_ok=True)

# 1) 域随机化 YAML
with open(DATA_DIR / 'domain_rand.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(domain_rand, f, sort_keys=False)
print(f'Generated domain randomization config → {DATA_DIR/"domain_rand.yaml"}')

# 2) 参考轨迹 NumPy
files_and_trajs = {
    'hover_traj.npy':      hover_traj,
    'circle_traj.npy':     circle_traj,
    'z_wave_traj.npy':     z_wave_traj,
    'accel_ref_traj.npy':  accel_ref_traj,   # 新增
}
for name, traj in files_and_trajs.items():
    path = REF_DIR / name
    np.save(path, traj)
    print(f'Generated reference trajectory → {path}  (shape={traj.shape})')
