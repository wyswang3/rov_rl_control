#!/usr/bin/env python3
"""
scripts/generate_ppo_config.py

生成 PPO 训练配置文件 ppo_accel_tuned.yaml（加速度控制任务，CPU 友好版）。

配置项说明：
  - env:      构造 ROVDynEnv 时需要的参数
  - reward:   各项奖励权重，传给环境
  - sampling: 并行环境数量与 batch_size
  - ppo:      PPO 算法本身的超参数
  - train:    训练计划（总步数、随机种子等）
"""

import yaml
from pathlib import Path

# ———— 配置分块 ————

# 1) 环境参数 —— 对应 ROVDynEnv.__init__ 中的前几个参数
env_cfg = {
    "dt":                 0.02,    # 时间步长 (s)
    "max_power":          60.0,    # 最大推进器功率 (W)
    "window_size":        9,       # LSTM 输入序列长度
    "accel_filter_alpha": 0.5,     # 低通滤波系数 (0~1)
}

# 2) 奖励权重 —— 传给 ROVDynEnv，让 step() 内使用
reward_cfg = {
    "w_err":  1.0,    # 加速度误差惩罚权重
    "w_jerk": 0.01,   # jerk 惩罚权重
    "w_eng":  0.001,  # 能耗惩罚权重
    # 如果未来需要，还可以添加 e.g. "w_att": 0.1
}

# 3) 并行与采样 —— 传给 train 脚本中用于创建 VecEnv
sampling_cfg = {
    "n_envs":     1,      # CPU 上通常设为 1；如果有 GPU 或多核可以改大
    "batch_size": 1024,   # PPO 总批量大小，内部会计算 n_steps = batch_size // n_envs
}

# 4) PPO 算法超参数 —— 直接传给 Stable-Baselines3 PPO 构造函数
ppo_cfg = {
    "learning_rate": 1e-4,   # PPO 学习率
    "gamma":         0.99,   # 折扣因子
    "gae_lambda":    0.95,   # GAE λ
    "ent_coef":      0.0,    # 熵系数
    "vf_coef":       0.5,    # 值函数损失系数
    "clip_range":    0.12,   # PPO 裁剪范围
    "clip_range_vf": 0.12,   # 对 value function 也做裁剪
    "max_grad_norm": 0.5,    # 梯度裁剪范数
    "n_epochs":      8,      # PPO 内循环更新次数
}

# 5) 训练计划 —— 总步数、随机种子
train_cfg = {
    "total_timesteps": 500_000,  # 总训练步数
    "seed":            42,       # 随机种子
    # 如果需要可再加 "save_interval": 50_000，来支持中途定期保存
}

# ———— 合并为最终配置 ————
config = {
    "env":      env_cfg,
    "reward":   reward_cfg,
    "sampling": sampling_cfg,
    "ppo":      ppo_cfg,
    "train":    train_cfg,
}

# ———— 写出 YAML ————
out_dir  = Path(__file__).resolve().parent / "configs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "ppo_accel_tuned.yaml"

with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(config, f, sort_keys=False, allow_unicode=True)

print(f"Generated tuned PPO config → {out_path}")
