#!/usr/bin/env python3
# envs/vector/make_vec_env.py

"""
Factory for creating a vectorized ROVDynEnv wrapped with VecMonitor and VecNormalize,
and configured to use GPU during model inference when available.

从 cfg 中读取：
  - cfg["env"]       → ROVDynEnv 的基本参数（traj_name、dt, max_power, window_size, accel_filter_alpha…）
  - cfg["reward"]    → ROVDynEnv 的奖励权重（k_pos, k_att, k_vel, k_jerk, k_eng, k_succ, pos_tol…）
  - cfg["sampling"]  → 并行环境数 n_envs 及 batch_size（仅用于并行数，不直接传给 Env）
  - cfg["train"]     → 随机种子 seed（用于环境可复现）
  - cfg["ppo"]       → PPO 算法的超参数（无需在这里处理）

最终返回的 VecEnv：DummyVecEnv 或 SubprocVecEnv → VecMonitor → VecNormalize
ROVDynEnv 内部会根据传入的 `device` 参数决定模型加载到 GPU 还是 CPU。
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)
from envs.rov_dyn_env import ROVDynEnv


__all__ = ["make_vec_env"]


def _build_env_kwargs(env_cfg, reward_cfg, device):
    env_kwargs = {
        # —— 环境基础参数 —— #
        "traj_name":         env_cfg.get("traj_name", "hover"),
        "dt":                env_cfg.get("dt", 0.02),
        "max_power":         env_cfg.get("max_power", 60.0),
        "device":            device,
        "window_size":       env_cfg.get("window_size", 9),
        "accel_filter_alpha": env_cfg.get("accel_filter_alpha", 0.5),

        # —— 奖励权重 —— #
        "k_pos":   reward_cfg.get("k_pos", 1.0),
        "k_att":   reward_cfg.get("k_att", 0.5),
        "k_vel":   reward_cfg.get("k_vel", 0.1),
        "k_jerk":  reward_cfg.get("k_jerk", 0.01),
        "k_acc":   reward_cfg.get("k_acc", 1.0),
        "k_eng":   reward_cfg.get("k_eng", 0.001),
        "k_succ":  reward_cfg.get("k_succ", 5.0),
        "pos_tol": reward_cfg.get("pos_tol", 0.1),
    }
    print("[DEBUG] env_kwargs in make_vec_env:", env_kwargs)
    return env_kwargs

def _make_single_env_fn(
    rank: int,
    seed: int,
    env_kwargs: Dict[str, Any]
) -> Callable[[], ROVDynEnv]:
    """
    返回一个可调用的函数，用于在 SubprocVecEnv 或 DummyVecEnv 中创建每个子环境实例。
    每个子环境会使用不同的 seed 以保证可复现性。
    """

    def _init_env() -> ROVDynEnv:
        env = ROVDynEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        return env

    return _init_env


def _choose_vector_env(
    env_fns: List[Callable[[], ROVDynEnv]],
    device: str,
    asynchronous: Optional[bool],
    n_envs: int
):
    """
    根据平台、device 和用户指定的 asynchronous 标志来决定使用 DummyVecEnv 还是 SubprocVecEnv。
    - Windows 平台统一使用 DummyVecEnv。
    - Linux+GPU 且 n_envs>1 且 asynchronous is None 时默认使用 SubprocVecEnv。
    - 如果 asynchronous 显式为 False，则使用 DummyVecEnv；为 True 则使用 SubprocVecEnv。
    """
    is_windows = sys.platform.startswith("win")
    if asynchronous is None:
        use_subproc = (not is_windows) and (device.startswith("cuda")) and (n_envs > 1)
    else:
        use_subproc = bool(asynchronous)

    if use_subproc:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


def make_vec_env(
    cfg: Dict[str, Any],
    device: str,
    n_envs: int,
    asynchronous: Optional[bool] = None
):
    """
    创建一个向量化的 ROVDynEnv，并依次套上 VecMonitor、VecNormalize。

    参数:
      - cfg          : 从 YAML 读取的完整配置 dict。
                       需包含 "env"、"reward"、"train" 三个子字典。
      - device       : "cpu" 或 "cuda"，传递给 ROVDynEnv 决定模型加载位置。
      - n_envs       : 并行环境数量。
      - asynchronous : 是否强制使用 SubprocVecEnv。若为 None 则根据平台/设备自动判断。

    返回:
      - 一个经 VecMonitor & VecNormalize 包装好的 VecEnv。
    """
    # 0) 提取 config 中的子字典
    env_cfg    = cfg.get("env", {})
    reward_cfg = cfg.get("reward", {})
    train_cfg  = cfg.get("train", {})

    # 1) 构建 ROVDynEnv 初始化需要的关键字参数
    env_kwargs = _build_env_kwargs(env_cfg, reward_cfg, device)

    # 2) 随机种子：从 train 部分读取 seed，若不存在则默认为 0
    base_seed = train_cfg.get("seed", 0)

    # 3) 构造每个子环境的工厂函数列表
    env_fns: List[Callable[[], ROVDynEnv]] = []
    for rank in range(n_envs):
        fn = _make_single_env_fn(rank, base_seed, env_kwargs)
        env_fns.append(fn)

    # 4) 根据条件选择 DummyVecEnv 或 SubprocVecEnv
    vec_env = _choose_vector_env(env_fns, device, asynchronous, n_envs)

    # 5) 包装 VecMonitor 用于监控 episode reward 和 length
    vec_env = VecMonitor(vec_env)

    # 6) 包装 VecNormalize 用于对观测做归一化, reward 不归一
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    return vec_env
