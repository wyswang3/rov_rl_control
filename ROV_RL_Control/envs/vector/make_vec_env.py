#!/usr/bin/env python3
# envs/vector/make_vec_env.py

"""
Factory for creating a vectorized ROVDynEnv, wrapped with VecMonitor and VecNormalize,
and configured to use GPU during model inference when available.

从 cfg 中读取：
  - cfg["env"]       → ROVDynEnv 的基本参数（dt, max_power, window_size, accel_filter_alpha…）
  - cfg["reward"]    → ROVDynEnv 的奖励权重（k_pos, k_att, k_vel, k_jerk, k_eng, k_succ, pos_tol…）
  - cfg["sampling"]  → 并行环境数 n_envs 及 batch_size
  - cfg["train"]     → 随机种子 seed, total_timesteps 等（若需）
  - cfg["ppo"]       → PPO 算法的超参数（若需）

最终返回 DummyVecEnv（或 SubprocVecEnv）→ VecMonitor → VecNormalize，
其中 ROVDynEnv 内部会根据传入的 `device` 参数在 GPU/CPU 上加载网络模型。
"""

import sys
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from envs.rov_dyn_env import ROVDynEnv


__all__ = ["make_vec_env"]


def make_vec_env(
    cfg: dict,
    device: str,
    n_envs: int,
    asynchronous: bool = None
):
    """
    创建一个向量化的 ROVDynEnv，并依次套上 VecMonitor, VecNormalize。

    参数:
      - cfg          : 从 YAML 读取的完整配置 dict
      - device       : "cpu" 或 "cuda"，传递给 ROVDynEnv 以决定模型加载的位置
      - n_envs       : 并行环境数量
      - asynchronous : 是否使用 SubprocVecEnv；若为 None，则在 Linux+GPU 环境下自动启用 SubprocVecEnv，
                      否则强制使用 DummyVecEnv。

    返回:
      - 一个被 VecMonitor + VecNormalize 包装好的 SB3 VectorEnv
    """

    # ── 1) 从 cfg 中提取 ROVDynEnv 构造时所需的参数 ──
    env_cfg    = cfg.get("env", {})
    reward_cfg = cfg.get("reward", {})

    # 将 cfg["env"] 中的 key/values 对应到 ROVDynEnv 初始化参数
    # 注意：我们要确保传了 device，让模型在 GPU 上运行
    env_kwargs = {
        "dt":                  env_cfg.get("dt", 0.02),
        "max_power":           env_cfg.get("max_power", 60.0),
        "device":              device,
        "window_size":         env_cfg.get("window_size", 9),
        "accel_filter_alpha":  env_cfg.get("accel_filter_alpha", 0.5),
        # 奖励权重
        "k_pos":               reward_cfg.get("k_pos", 1.0),
        "k_att":               reward_cfg.get("k_att", 0.5),
        "k_vel":               reward_cfg.get("k_vel", 0.1),
        "k_jerk":              reward_cfg.get("k_jerk", 0.01),
        "k_eng":               reward_cfg.get("k_eng", 0.001),
        "k_succ":              reward_cfg.get("k_succ", 5.0),
        "pos_tol":             reward_cfg.get("pos_tol", 0.1),
    }

    # ── 2) 定义“工厂函数”——每个子环境的初始化：──
    def make_single_env(rank: int):
        def _init():
            env = ROVDynEnv(**env_kwargs)
            # 给每个子环境一个不同的 seed（rank），保持可复现性
            env.reset(seed=rank)
            return env
        return _init

    env_fns = [make_single_env(i) for i in range(n_envs)]

    # ── 3) 选择使用 DummyVecEnv 还是 SubprocVecEnv ──
    is_windows = sys.platform.startswith("win")
    # 如果未指定 asynchronous，就在 Linux+GPU 下默认启用 SubprocVecEnv；否则按用户要求
    if asynchronous is None:
        use_subproc = (not is_windows) and (device == "cuda") and (n_envs > 1)
    else:
        use_subproc = asynchronous

    if use_subproc:
        vec = SubprocVecEnv(env_fns)
    else:
        vec = DummyVecEnv(env_fns)

    # ── 4) 再套上 VecMonitor 和 VecNormalize ──
    # VecMonitor 用于记录 episode reward/length；VecNormalize 会对 obs 做归一化
    vec = VecMonitor(vec)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)

    return vec
