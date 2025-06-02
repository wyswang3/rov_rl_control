#!/usr/bin/env python3
"""
scripts/evaluate_with_norm.py

加载训练时保存的 VecNormalize 与 PPO 模型，
对若干条 rollout 跑完后剔除暖机阶段误差，打印并绘制加速度跟踪误差曲线。
"""

import sys
import os
# 把项目根目录加入模块搜索路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from pathlib import Path
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from envs.vector.make_vec_env import make_vec_env


def load_config(cfg_path: str) -> dict:
    """读取并返回 YAML 配置字典。"""
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_eval_env(cfg: dict, norm_path: str) -> VecNormalize:
    """
    创建并返回一个已加载 VecNormalize 状态的评估环境。
    - cfg: 配置字典
    - norm_path: VecNormalize 保存文件 (.pkl)
    """
    base_env = make_vec_env(
        cfg,
        device='cpu',
        n_envs=1,
        asynchronous=False
    )
    vec_env = VecNormalize.load(norm_path, base_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def load_policy(model_path: str, env: VecNormalize) -> PPO:
    """
    从文件加载 PPO 策略并绑定到给定环境上。
    """
    return PPO.load(model_path, env=env, device='cpu')


def run_rollouts(
    model: PPO,
    env: VecNormalize,
    num_eps: int,
    horizon: int,
    warmup_steps: int
) -> np.ndarray:
    """
    对环境做多次滚动评估，收集每步加速度误差并剔除前 warmup_steps。
    返回 shape = (num_eps, <=horizon - warmup_steps) 的误差数组列表（已填充）。
    """
    all_errs = []

    for ep in range(num_eps):
        obs = env.reset()  # shape (1, obs_dim)
        errs = []

        for t in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            accel = infos[0].get('accel', np.zeros(6, dtype=np.float32))
            errs.append(np.linalg.norm(accel))  # 目标为零

            if dones[0]:
                break

        # 剔除暖机阶段
        trimmed = errs[warmup_steps:] if len(errs) > warmup_steps else []
        all_errs.append(trimmed)

    # 统一长度并转换为 array
    max_len = max(len(e) for e in all_errs) if all_errs else 0
    padded = np.array([
        np.pad(e, (0, max_len - len(e)), 'edge')
        for e in all_errs
    ], dtype=np.float32)

    return padded


def summarize_errors(errs: np.ndarray) -> None:
    """
    打印每条 episode 的平均误差与总体平均误差。
    """
    per_ep = errs.mean(axis=1)
    overall = errs.mean()
    print(f"Mean error per episode: {per_ep}")
    print(f"Overall mean error     : {overall:.6f}")


def plot_errors(
    errs: np.ndarray,
    dt: float,
    out_path: Path
) -> None:
    """
    绘制加速度误差的均值 ±1σ 曲线并保存到 out_path。
    """
    mean_err = errs.mean(axis=0)
    std_err  = errs.std(axis=0)
    t = np.arange(len(mean_err)) * dt

    plt.figure(figsize=(8, 4))
    plt.fill_between(t, mean_err - std_err, mean_err + std_err,
                     alpha=0.3, label='±1 std')
    plt.plot(t, mean_err, label='mean accel error')
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m/s²)")
    plt.title("Acceleration Tracking Error (post-warmup)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot → {out_path}")


def main():
    # --------------- CLI ---------------
    parser = argparse.ArgumentParser(
        description="Evaluate PPO policy with VecNormalize"
    )
    parser.add_argument("--model",  required=True,
                        help="路径到 PPO 模型文件 (.zip)")
    parser.add_argument("--norm",   required=True,
                        help="路径到 VecNormalize 文件 (.pkl)")
    parser.add_argument("--cfg",    default="trainer/configs/ppo_accel.yaml",
                        help="训练时使用的 config 文件路径")
    parser.add_argument("--eps",    type=int, default=3,
                        help="评估 rollout 次数")
    parser.add_argument("--warmup", type=float, default=1.0,
                        help="暖机时长（秒），将剔除前面这段误差）")
    args = parser.parse_args()

    # --------------- 准备 ---------------
    cfg = load_config(args.cfg)
    dt = cfg['env']['dt']
    horizon = int(cfg['env'].get('eval_horizon', 1000))
    warmup_steps = int(args.warmup / dt)

    # 把项目根加入 path（确保能 import envs/）
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    # --------------- 构建环境 & 策略 ---------------
    env   = build_eval_env(cfg, args.norm)
    model = load_policy(args.model, env)

    # --------------- 评估 ---------------
    errs = run_rollouts(model, env, args.eps, horizon, warmup_steps)

    # --------------- 结果 ---------------
    summarize_errors(errs)
    plot_errors(errs, dt, Path("eval_with_norm.png"))


if __name__ == "__main__":
    main()
