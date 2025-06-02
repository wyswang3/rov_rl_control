#!/usr/bin/env python3
"""
scripts/evaluate_with_norm.py

加载训练时保存的 VecNormalize 与 PPO 模型，
对若干条 rollout 跑完后剔除暖机阶段误差，打印并绘制加速度跟踪误差曲线。

Usage 示例（假设当前路径为项目根即含有 envs/、trainer/ 等子目录）：
  python scripts/evaluate_with_norm.py \
    --model runs/ppo/20250602_203057/checkpoints/ppo_final.zip \
    --norm  runs/ppo/20250602_203057/checkpoints/vec_normalize.pkl \
    --cfg   trainer/configs/ppo_accel_tuned.yaml \
    --eps   5 \
    --warmup 1.0
"""

import sys
import os
from pathlib import Path
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

# ─── 在任何 import envs.* 之前，都先把项目根目录加到 sys.path ───
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# 现在可以安全地导入 make_vec_env 了
from envs.vector.make_vec_env import make_vec_env


def load_config(cfg_path: str) -> dict:
    """读取并返回 YAML 配置字典。"""
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_eval_env(cfg: dict, norm_path: str) -> VecNormalize:
    """
    创建并返回一个已加载 VecNormalize 状态的评估环境。
    - cfg: 配置字典（包含 'env'、'reward'、'sampling' 等块）
    - norm_path: VecNormalize 保存文件 (.pkl)
    """
    # “基础环境”：n_envs=1，同步环境
    base_env = make_vec_env(
        cfg,
        device='cpu',
        n_envs=1,
        asynchronous=False
    )
    # 加载训练时保存下来的 VecNormalize 参数
    vec_env = VecNormalize.load(norm_path, base_env)
    # 评估时不更新归一化统计，也不归一化 reward
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
    对环境做多次 rollout 评估，收集每步加速度误差并剔除前 warmup_steps。
    返回 shape = (num_eps, <= horizon - warmup_steps) 的误差数组列表（已对齐并填充）。
    """
    all_errs = []

    for ep in range(num_eps):
        # reset() 返回 obs 数组，形状 (1, obs_dim)
        obs = env.reset()
        errs = []

        for t in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)

            # 从 info 中取“滤波后”的线性+角加速度（6 维）
            accel = infos[0].get('accel', np.zeros(6, dtype=np.float32))
            # 目标为零，所以误差 = ||accel||₂
            errs.append(np.linalg.norm(accel))

            if dones[0]:
                break

        # 剔除暖机阶段
        if len(errs) > warmup_steps:
            trimmed = errs[warmup_steps:]
        else:
            trimmed = []

        all_errs.append(trimmed)

    # 对各条 episode list 进行“右侧填充”，使长度一致
    max_len = max((len(e) for e in all_errs), default=0)
    padded = np.array([
        np.pad(e, (0, max_len - len(e)), 'edge') if len(e) < max_len else np.array(e)
        for e in all_errs
    ], dtype=np.float32)

    return padded  # shape = (num_eps, max_len)


def summarize_errors(errs: np.ndarray) -> None:
    """
    打印每条 episode 的平均误差与总体平均误差。
    """
    if errs.size == 0:
        print("No data collected.")
        return

    per_ep_mean = errs.mean(axis=1)
    overall_mean = errs.mean()
    print("Mean error per episode:", per_ep_mean)
    print(f"Overall mean error     : {overall_mean:.6f}")


def plot_errors(
    errs: np.ndarray,
    dt: float,
    out_path: Path
) -> None:
    """
    绘制加速度误差的均值 ±1σ 曲线并保存到 out_path。
    """
    if errs.size == 0:
        print("No data to plot.")
        return

    mean_err = errs.mean(axis=0)
    std_err  = errs.std(axis=0)
    t = np.arange(len(mean_err)) * dt

    plt.figure(figsize=(8, 4))
    plt.fill_between(
        t, mean_err - std_err, mean_err + std_err,
        alpha=0.3, label='±1 σ'
    )
    plt.plot(t, mean_err, label='Mean acceleration error')
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m/s²)")
    plt.title("Acceleration Tracking Error (post-warmup)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot → {out_path}")


def main():
    # ------------------- 1) 解析 CLI 参数 -------------------
    parser = argparse.ArgumentParser(
        description="Evaluate PPO policy with VecNormalize"
    )
    parser.add_argument(
        "--model", required=True,
        help="路径到 PPO 模型文件 (.zip)"
    )
    parser.add_argument(
        "--norm", required=True,
        help="路径到 VecNormalize 文件 (.pkl)"
    )
    parser.add_argument(
        "--cfg", default="trainer/configs/ppo_accel_tuned.yaml",
        help="训练时使用的 config 文件路径"
    )
    parser.add_argument(
        "--eps", type=int, default=3,
        help="评估 rollout 次数"
    )
    parser.add_argument(
        "--warmup", type=float, default=1.0,
        help="暖机时长（秒），将剔除前面这段误差）"
    )
    args = parser.parse_args()

    # ------------------- 2) 读取配置 -------------------
    cfg = load_config(args.cfg)
    dt = cfg['env']['dt']
    # 如果配置里有 'eval_horizon'，就优先用它，否则默认 1000 步
    horizon = int(cfg['env'].get('eval_horizon', 1000))
    warmup_steps = int(args.warmup / dt)

    # ------------------- 3) 构建评估环境 & 加载策略 -------------------
    env   = build_eval_env(cfg, args.norm)
    model = load_policy(args.model, env)

    # ------------------- 4) 运行若干次 rollout，收集误差 -------------------
    errs = run_rollouts(model, env, args.eps, horizon, warmup_steps)

    # ------------------- 5) 打印 & 绘图 -------------------
    summarize_errors(errs)
    plot_errors(errs, dt, Path("eval_with_norm.png"))


if __name__ == "__main__":
    main()
