#!/usr/bin/env python3
"""
trainer/configs/generate_config.py

生成 PPO 训练配置文件 ppo_accel_tuned.yaml（位置+姿态+加速度 控制任务，GPU/CPU 共用）。

使用示例：
    # 默认生成 hover 轨迹的配置：
    $ python generate_config.py

    # 指定其他轨迹、并行环境数、训练步数等：
    $ python generate_config.py \
        --traj_name constant_acc \
        --n_envs 2 \
        --total_timesteps 1000000 \
        --output_path trainer/configs/ppo_custom.yaml
"""

import argparse
import yaml
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，用于覆盖默认配置。
    """
    parser = argparse.ArgumentParser(
        description="生成并输出 PPO 训练配置（ppo_accel_tuned.yaml）"
    )

    # —— 环境（env）相关参数 —— #
    parser.add_argument(
        "--traj_name",
        type=str,
        default="hover",
        choices=[
            "hover",
            "circle",
            "z_wave",
            "ptp_forward",
            "inspect_pipe",
            "const_acc",
        ],
        help="参考轨迹名称，对应 data/ref_trajs/{traj_name}_traj.npy (增加 'constant_acc')"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="仿真环境的时间步长（秒）"
    )
    parser.add_argument(
        "--max_power",
        type=float,
        default=160.0,
        help="最大推进器功率（瓦特）"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=9,
        help="LSTM 序列长度（帧数）"
    )
    parser.add_argument(
        "--accel_filter_alpha",
        type=float,
        default=0.5,
        help="加速度低通滤波系数 α ∈ [0,1]"
    )

    # —— 奖励（reward）相关参数 —— #
    parser.add_argument("--k_pos",   type=float, default=0.0,   help="位置误差惩罚系数")
    parser.add_argument("--k_att",   type=float, default=0.001, help="姿态误差惩罚系数")
    parser.add_argument("--k_vel",   type=float, default=0.1,  help="速度惩罚系数")
    parser.add_argument("--k_jerk",  type=float, default=0.01, help="角加速度变化惩罚系数")
    parser.add_argument("--k_eng",   type=float, default=0.001, help="能耗惩罚系数")
    parser.add_argument("--k_acc",   type=float, default=0.1, help="加速度跟踪误差惩罚系数")
    parser.add_argument("--k_succ",  type=float, default=5.0,   help="成功到达目标奖励")
    parser.add_argument("--pos_tol", type=float, default=0.1,   help="到达目标的容忍距离（米）")

    # —— 采样（sampling）相关参数 —— #
    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,
        help="并行环境数量"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="总 batch size（内部 n_steps = batch_size // n_envs）"
    )

    # —— PPO 算法超参数 —— #
    parser.add_argument(
        "--initial_lr",
        type=float,
        default=1e-4,
        help="初始学习率（线性衰减）"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="折扣因子 γ"
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="GAE λ"
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.0,
        help="熵系数"
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=0.5,
        help="值函数损失系数"
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=0.12,
        help="PPO 剪切范围 ε"
    )
    parser.add_argument(
        "--clip_range_vf",
        type=float,
        default=0.12,
        help="值函数剪切范围 ε_vf"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="梯度裁剪阈值"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=8,
        help="PPO 内循环更新次数"
    )

    # —— 训练计划（train）相关参数 —— #
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=500_000,
        help="总训练步数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    # —— 输出路径 —— #
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(__file__).resolve().parent / "ppo_accel_tuned.yaml",
        help="生成的 YAML 文件保存路径"
    )

    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    构建 env 配置字典。
    """
    return {
        "traj_name":          args.traj_name,
        "dt":                 args.dt,
        "max_power":          args.max_power,
        "window_size":        args.window_size,
        "accel_filter_alpha": args.accel_filter_alpha,
    }


def build_reward_config(args: argparse.Namespace) -> Dict[str, float]:
    """
    构建 reward 配置字典。
    """
    return {
        "k_pos":   args.k_pos,
        "k_att":   args.k_att,
        "k_vel":   args.k_vel,
        "k_jerk":  args.k_jerk,
        "k_eng":   args.k_eng,
        "k_acc":   args.k_acc,
        "k_succ":  args.k_succ,
        "pos_tol": args.pos_tol,
    }


def build_sampling_config(args: argparse.Namespace) -> Dict[str, int]:
    """
    构建 sampling 配置字典。
    """
    return {
        "n_envs":     args.n_envs,
        "batch_size": args.batch_size,
    }


def build_ppo_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    构建 PPO 算法超参数配置字典。
    """
    return {
        "learning_rate": {
            "type":       "linear",
            "initial_lr": args.initial_lr,
        },
        "gamma":         args.gamma,
        "gae_lambda":    args.gae_lambda,
        "ent_coef":      args.ent_coef,
        "vf_coef":       args.vf_coef,
        "clip_range":    args.clip_range,
        "clip_range_vf": args.clip_range_vf,
        "max_grad_norm": args.max_grad_norm,
        "n_epochs":      args.n_epochs,
    }


def build_train_config(args: argparse.Namespace) -> Dict[str, int]:
    """
    构建 train 配置字典。
    """
    return {
        "total_timesteps": args.total_timesteps,
        "seed":            args.seed,
    }


def write_yaml(config: Dict[str, Any], output_path: Path) -> None:
    """
    将完整配置写出到 YAML 文件。
    """
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] Generated PPO config → {output_path.resolve()}")


def main():
    args = parse_args()

    # 构建各模块配置
    env_cfg      = build_env_config(args)
    reward_cfg   = build_reward_config(args)
    sampling_cfg = build_sampling_config(args)
    ppo_cfg      = build_ppo_config(args)
    train_cfg    = build_train_config(args)

    # 合并成最终配置字典
    config = {
        "env":      env_cfg,
        "reward":   reward_cfg,
        "sampling": sampling_cfg,
        "ppo":      ppo_cfg,
        "train":    train_cfg,
    }

    # 写出 YAML
    write_yaml(config, args.output_path)


if __name__ == "__main__":
    main()
