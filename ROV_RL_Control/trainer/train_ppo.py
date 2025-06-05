#!/usr/bin/env python3
"""
trainer/train_ppo.py

PPO 训练脚本（使用 envs/vector/make_vec_env 中的 make_vec_env）：
  1. 读取 YAML 配置
  2. 自动选择 CPU/GPU
  3. 在 rollout 结束后，将 Returns/Advantages 统计记录到 TensorBoard
  4. 保存训练日志到文件，以便事后分析
  5. 保存最终模型与 VecNormalize 参数
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import yaml
import torch
import numpy as np
from typing import Any, Dict, Union, Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.vector.make_vec_env import make_vec_env


class ROVTensorboardCallback(BaseCallback):
    """
    自定义 TensorBoard 回调， 在每个 rollout 结束时记录 returns 和 advantages 的均值与标准差。
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 每一步都返回 True，表示继续训练
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        returns = buf.returns.flatten()
        advs = buf.advantages.flatten()
        self.logger.record("rollout/returns_mean", float(returns.mean()))
        self.logger.record("rollout/returns_std", float(returns.std()))
        self.logger.record("rollout/adv_mean", float(advs.mean()))
        self.logger.record("rollout/adv_std", float(advs.std()))


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，仅有一个 --config 字段用于指定 YAML 配置文件路径。
    """
    parser = argparse.ArgumentParser(description="Train PPO on ROVDynEnv")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "ppo_accel_tuned.yaml",
        help="PPO 配置文件路径 (YAML)"
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    """
    从给定路径读取 YAML 配置，返回字典。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_random_seed(seed: int) -> None:
    """
    设置 NumPy 和 PyTorch 的随机种子，确保可复现。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dirs(base: Path) -> tuple:
    """
    在 runs/ppo/<timestamp> 下创建 checkpoints、tensorboard 与 logs 子目录：
      - checkpoints: 用于保存最终模型
      - tensorboard: 用于 TensorBoard 日志
      - logs: 用于保存训练过程中打印的日志
    返回 (ckpt_dir, tb_dir, log_file_path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = base / "runs" / "ppo" / timestamp
    ckpt_dir = run_root / "checkpoints"
    tb_dir = run_root / "tensorboard"
    log_dir = run_root / "logs"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / "training.log"
    return ckpt_dir, tb_dir, log_file_path


def build_learning_rate(lr_cfg: Any) -> Union[float, Callable[[float], float]]:
    """
    处理配置中的 learning_rate 字段：
      - 如果是 float，直接返回该值
      - 如果是 dict 并且 type == 'linear'，返回一个线性衰减的函数
    """
    if isinstance(lr_cfg, dict):
        lr_type = lr_cfg.get("type", "linear")
        init_lr = float(lr_cfg.get("initial_lr", 1e-4))
        if lr_type == "linear":
            return lambda progress_remaining: init_lr * progress_remaining
        else:
            raise ValueError(f"Unsupported learning_rate type: {lr_type}")
    else:
        return float(lr_cfg)


def build_model(
    cfg: Dict,
    vec_env,
    tb_dir: Path,
    device: str
) -> PPO:
    """
    根据配置构建并返回 PPO 模型实例：
      - policy: 固定 'MlpPolicy'
      - env: vec_env
      - learning_rate: 可能是常数或 callable
      - n_steps: batch_size // n_envs
      - 其余参数直接从 cfg['ppo'] 中读取
    """
    sampling = cfg.get("sampling", {})
    ppo_cfg = cfg.get("ppo", {})

    n_envs = sampling.get("n_envs", 4)
    batch_size = sampling.get("batch_size", 1024)
    n_steps = batch_size // n_envs

    lr = build_learning_rate(ppo_cfg.get("learning_rate", 3e-4))
    gamma = ppo_cfg.get("gamma", 0.99)
    gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
    ent_coef = ppo_cfg.get("ent_coef", 0.0)
    vf_coef = ppo_cfg.get("vf_coef", 0.5)
    clip_range = ppo_cfg.get("clip_range", 0.2)
    clip_range_vf = ppo_cfg.get("clip_range_vf", None)
    max_grad_norm = ppo_cfg.get("max_grad_norm", None)
    n_epochs = ppo_cfg.get("n_epochs", 4)

    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr,
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


def setup_logging(log_file: Path) -> None:
    """
    配置 logging，将 INFO 及以上级别的日志同时输出到控制台和文件 log_file。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 文件处理器
    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    logging.info("Logging initialized. Writing to %s", log_file)


def main(config_path: Path) -> None:
    # 1) 读取 YAML 配置
    cfg = load_config(config_path)

    # 2) 选择设备：CUDA 优先，否则 CPU
    # —— 硬编码想用的 GPU 索引 —— #
    PREFERRED_GPU = 2  # 索引从 0 开始：0,1,2,3…，这里表示“第 3 张卡”

    if torch.cuda.is_available() and torch.cuda.device_count() > PREFERRED_GPU:
        device = f"cuda:{PREFERRED_GPU}"
    else:
        device = "cpu"
    logging.basicConfig(level=logging.INFO)  # 临时基础配置
    logging.info(f"[INFO] Device: {device}")

    # 3) 提取配置块
    sampling_params = cfg.get("sampling", {})
    train_params = cfg.get("train", {})

    n_envs = sampling_params.get("n_envs", 1)
    total_timesteps = train_params.get("total_timesteps", 200_000)
    seed = train_params.get("seed", None)

    # 4) 设置随机种子（可选）
    if seed is not None:
        set_random_seed(seed)
        logging.info(f"[INFO] Random seed set to {seed}")

    # 5) 创建输出目录（包含日志文件）
    ckpt_dir, tb_dir, log_file = create_dirs(Path("."))
    # 先重新配置 logging，写入刚创建的 log_file
    logging.getLogger().handlers.clear()
    setup_logging(log_file)

    # 6) 创建向量化环境
    logging.info(f"[INFO] Creating vectorized env: n_envs = {n_envs}")
    vec_env = make_vec_env(cfg, device, n_envs, asynchronous=False)
    if seed is not None and hasattr(vec_env, "seed"):
        vec_env.seed(seed)

    # 7) 构建 PPO 模型
    model = build_model(cfg, vec_env, tb_dir, device)
    logging.info(f"[INFO] Starting training for {total_timesteps} timesteps")

    # 8) 训练并记录 TensorBoard、必要时记录到日志文件
    callback = ROVTensorboardCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # 9) 保存模型
    final_path = ckpt_dir / "ppo_final"
    model.save(str(final_path))
    logging.info(f"[INFO] Model saved to {final_path}.zip")

    # 10) 如果使用 VecNormalize，保存归一化参数
    try:
        vec_env.save(str(ckpt_dir / "vec_normalize.pkl"))
        logging.info(f"[INFO] VecNormalize stats saved to {ckpt_dir/'vec_normalize.pkl'}")
    except Exception:
        pass

    # 11) 关闭环境
    vec_env.close()
    logging.info("[INFO] Training complete, environment closed.")


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
