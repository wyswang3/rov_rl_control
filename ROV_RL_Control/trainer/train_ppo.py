#!/usr/bin/env python3
"""
trainer/train_ppo.py

PPO 训练脚本（使用 envs/vector/make_vec_env.py 中的 make_vec_env）：
- 读取 YAML 配置
- 自动选择 CPU/GPU
- 在 rollout 结束后，将 Returns/Advantages 统计记录到 TensorBoard
- 保存最终模型与 VecNormalize 参数
"""

import sys
import yaml
import torch
import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# 注意：务必使用 envs.vector.make_vec_env（相对于项目根目录）
from envs.vector.make_vec_env import make_vec_env


class ROVTensorboardCallback(BaseCallback):
    """
    在 rollout 结束后，把 returns 与 advantages 的均值/标准差记录到 TensorBoard。
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 每个 step 都返回 True，表示继续训练
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        returns    = buf.returns.flatten()
        advantages = buf.advantages.flatten()
        self.logger.record("rollout/returns_mean", float(returns.mean()))
        self.logger.record("rollout/returns_std",  float(returns.std()))
        self.logger.record("rollout/adv_mean",     float(advantages.mean()))
        self.logger.record("rollout/adv_std",      float(advantages.std()))


def main(config_path: Path):
    # 1) 读取 YAML 配置
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2) 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 3) 从 cfg 中提取各项参数
    #    - env_params 传递给 ROVDynEnv
    #    - reward_params 也传递给 ROVDynEnv
    #    - sampling_params、ppo_params、train_params 分别传给采样、算法、训练计划
    env_params      = cfg.get("env", {})      # ROVDynEnv 需要的 env 参数
    reward_params   = cfg.get("reward", {})   # ROVDynEnv 需要的 reward 参数
    sampling_params = cfg.get("sampling", {})
    ppo_params      = cfg.get("ppo", {})
    train_params    = cfg.get("train", {})

    # 并行环境数量 & batch_size
    n_envs     = int(sampling_params.get("n_envs", 1))
    batch_size = int(sampling_params.get("batch_size", 512))

    # 训练总步数 & 随机种子
    total_timesteps = int(train_params.get("total_timesteps", 200_000))
    seed            = train_params.get("seed", None)

    # 4) 如果用户指定了 seed，就统一设定 PyTorch / NumPy 的随机种子
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Env 的随机种子会在 make_vec_env 内部由每个子环境 reset(seed=i) 来设置

    # 5) 创建向量化环境
    print(f"[INFO] Creating vectorized env: n_envs = {n_envs}")
    vec_env = make_vec_env(
        cfg,        # 【①】完整的配置字典 (make_vec_env 会内部读取 cfg['env']、cfg['reward'] 等)
        device,     # 【②】'cpu' 或 'cuda'
        n_envs,     # 【③】并行环境数量
        False       # 【④】asynchronous=False → 强制使用 DummyVecEnv（单进程）
    )

    # 如果 make_vec_env 返回的 VecEnv 支持 seed()，就统一设置一下
    if (seed is not None) and hasattr(vec_env, "seed"):
        vec_env.seed(seed)

    # 6) 准备输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root  = Path("runs") / "ppo" / timestamp
    ckpt_dir  = run_root / "checkpoints"
    tb_dir    = run_root / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # 7) 构建 PPO；n_steps = batch_size // n_envs
    n_steps       = batch_size // n_envs
    learning_rate = float(ppo_params.get("learning_rate", 3e-4))
    gamma         = float(ppo_params.get("gamma", 0.99))
    gae_lambda    = float(ppo_params.get("gae_lambda", 0.95))
    ent_coef      = float(ppo_params.get("ent_coef", 0.0))
    vf_coef       = float(ppo_params.get("vf_coef", 0.5))
    clip_range    = float(ppo_params.get("clip_range", 0.2))
    clip_range_vf = ppo_params.get("clip_range_vf", None)
    max_grad_norm = ppo_params.get("max_grad_norm", None)
    n_epochs      = int(ppo_params.get("n_epochs", 4))

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log=str(tb_dir),
        verbose=1,
        device=device,
    )
    print(f"[INFO] Starting training for {total_timesteps} timesteps")

    # 8) 训练并记录 Returns / Advantages 到 TensorBoard
    callback = ROVTensorboardCallback()
    # 注意：不要再传 seed=seed，因为 SB3 的 learn() 已不接受 seed 参数
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # 9) 保存最终模型
    final_model_path = ckpt_dir / "ppo_final"
    model.save(str(final_model_path))
    print(f"[INFO] Model saved to {final_model_path}.zip")

    # 10) 如果使用了 VecNormalize，就把统计信息也保存下来
    try:
        # VecNormalize.save() 会把mean/std写到指定pkl
        vec_env.save(str(ckpt_dir / "vec_normalize.pkl"))
        print(f"[INFO] VecNormalize stats saved to {ckpt_dir / 'vec_normalize.pkl'}")
    except Exception:
        # 如果 vec_env 不是 VecNormalize，就跳过
        pass

    # 11) 关闭环境
    vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on ROVDynEnv")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=(Path(__file__).resolve().parent / "configs" / "ppo_accel_tuned.yaml"),
        help="Path to PPO config file",
    )
    args = parser.parse_args()
    main(args.config)
