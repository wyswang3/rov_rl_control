#!/usr/bin/env python3
"""
envs/vector/make_vec_env.py

Factory for creating a vectorized ROV dynamics environment. Uses SB3’s
DummyVecEnv or SubprocVecEnv so that VecMonitor/VecNormalize can work properly.

Usage:
    vec_env = make_vec_env(
        cfg=cfg_dict,
        device="cpu",
        n_envs=4,
        asynchronous=False
    )
"""

import sys
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)
from envs.rov_dyn_env import ROVDynEnv


__all__ = ["make_vec_env"]


def make_vec_env(
    cfg: dict,
    device: str,
    n_envs: int,
    asynchronous: bool = False
):
    """
    Create a vectorized ROVDynEnv wrapped with VecMonitor and VecNormalize.

    Args:
        cfg (dict): Full configuration dictionary loaded from YAML. Expects keys:
            - "env": containing dt, max_power, window_size, accel_filter_alpha, etc.
            - "reward": containing w_err, w_jerk, w_eng, etc.
        device (str): "cpu" or "cuda"
        n_envs (int): Number of parallel environments to create
        asynchronous (bool): If True, use SubprocVecEnv (multiple processes).
                             Otherwise (or on Windows, or n_envs == 1), use DummyVecEnv.

    Returns:
        VecNormalize: A vectorized environment (VecMonitor + VecNormalize).
    """
    # 1) Extract environment-specific parameters from cfg
    env_cfg = cfg.get("env", {})
    reward_cfg = cfg.get("reward", {})

    env_kwargs = {
        "dt":                 env_cfg.get("dt", 0.02),
        "max_power":          env_cfg.get("max_power", 80.0),
        "window_size":        env_cfg.get("window_size", 9),
        "accel_filter_alpha": env_cfg.get("accel_filter_alpha", 0.5),
        "w_err":              reward_cfg.get("w_err", 1.0),
        "w_jerk":             reward_cfg.get("w_jerk", 0.5),
        "w_eng":              reward_cfg.get("w_eng", 0.01),
    }

    # 2) Define a factory for each sub‐environment
    def make_single_env(seed: int):
        def _init():
            env = ROVDynEnv(device=device, **env_kwargs)
            env.reset(seed=seed)
            return env
        return _init

    env_fns = [make_single_env(i) for i in range(n_envs)]

    # 3) Choose DummyVecEnv or SubprocVecEnv
    is_windows = sys.platform.startswith("win")
    if is_windows or not asynchronous or n_envs == 1:
        vec = DummyVecEnv(env_fns)
    else:
        vec = SubprocVecEnv(env_fns)

    # 4) Wrap with VecMonitor and VecNormalize
    vec = VecMonitor(vec)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)

    return vec
