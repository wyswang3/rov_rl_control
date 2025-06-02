#!/usr/bin/env python3
"""
trainer/configs/generate_config.py

生成 PPO 训练配置文件 ppo_accel_tuned.yaml（位置+姿态+加速度 控制任务，GPU/CPU 共用）。
这个脚本会把 YAML 写到：
    rov_rl_control/trainer/configs/ppo_accel_tuned.yaml
"""

import yaml
from pathlib import Path

# ———— 1) 环境参数 ————
env_cfg = {
    "dt":                 0.02,   # 时间步长 (s)
    "max_power":          60.0,   # 最大推进器功率 (W)
    "window_size":        9,      # LSTM 序列长度
    "accel_filter_alpha": 0.5,    # 加速度低通滤波系数
}

# ———— 2) 奖励权重 ————
reward_cfg = {
    "k_pos":   1.0,    # 位置误差惩罚系数
    "k_att":   0.5,    # 姿态误差惩罚系数
    "k_vel":   0.1,    # 速度惩罚系数
    "k_jerk":  0.01,   # 角加速度变化惩罚系数
    "k_eng":   0.001,  # 能耗惩罚系数
    "k_succ":  5.0,    # 成功到达目标的奖励
    "pos_tol": 0.1,    # 到达目标的容忍距离 (m)
}

# ———— 3) 并行与采样 ————
sampling_cfg = {
    "n_envs":     4,      # 并行环境数量（单张 GPU 或 CPU 测试时一般写 1）
    "batch_size": 1024,   # 总 batch size
    # 内部 n_steps = batch_size // n_envs
}

# ———— 4) PPO 算法超参数 ————
ppo_cfg = {
    "learning_rate": {
        "type":       "linear",   # 线性衰减
        "initial_lr": 1e-4,        # 初始学习率
    },
    "gamma":         0.99,    # 折扣因子
    "gae_lambda":    0.95,    # GAE λ
    "ent_coef":      0.0,     # 熵系数
    "vf_coef":       0.5,     # 值函数损失系数
    "clip_range":    0.12,    # PPO 剪切范围
    "clip_range_vf": 0.12,    # 值函数剪切（可选）
    "max_grad_norm": 0.5,     # 梯度裁剪
    "n_epochs":      8,       # PPO 内循环更新次数
}

# ———— 5) 训练计划 ————
train_cfg = {
    "total_timesteps": 500_000,  # 总训练步数
    "seed":            42,       # 随机种子
    # 如果在 Linux 上使用多 GPU/混合精度，可在此扩展更多字段
}

# ———— 合并配置 ————
config = {
    "env":      env_cfg,
    "reward":   reward_cfg,
    "sampling": sampling_cfg,
    "ppo":      ppo_cfg,
    "train":    train_cfg,
}

# ———— 写出到 trainer/configs/ ————
# Path(__file__).parent 就是 <项目根>/trainer/configs
out_dir  = Path(__file__).resolve().parent
out_path = out_dir / "ppo_accel_tuned.yaml"

# 确保目录存在
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(config, f, sort_keys=False, allow_unicode=True)

print(f"Generated tuned PPO config → {out_path.resolve()}")
