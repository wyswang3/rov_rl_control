#!/usr/bin/env python3
# tests/test_lstm_feedback_compare.py

"""
测试 LSTM-based HybridDynamicsModel 在两种 IMU 缓冲策略下的输入输出差异：
  A. 始终让 IMU 缓冲全零
  B. 用上一时刻模型的加速度输出逐步“反馈”到 IMU 缓冲

相同的随机 pw 序列下，比较 6 维加速度预测随时间的变化，并绘制对比图。
"""

import argparse
import sys
from pathlib import Path
from collections import deque

import torch
import numpy as np
import matplotlib.pyplot as plt

# ─── 保证项目根目录在 sys.path 中，以便找到 models.lstm_dyn.loader ─────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # 假设 tests/ 就在项目根目录下
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ────────────────────────────────────────────────────────────────────────────────────

from models.lstm_dyn.loader import load_dynamics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LSTM predictions with zero-IMU vs. feedback-IMU"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device，例如 'cpu' 或 'cuda:0'"
    )
    parser.add_argument(
        "--window_size", type=int, default=9,
        help="LSTM 序列长度（帧数）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    W = args.window_size
    N = W * 2  # 运行 2 * window_size 步

    print(f"[INFO] 使用设备 = {device}，window_size = {W}, 共评估 {N} 步\n")

    # 1) 加载预训练模型
    model, init_hidden_fn = load_dynamics(device)
    model.to(device)
    model.eval()
    print("[INFO] 成功加载 LSTM 动力学模型。\n")

    # 2) 随机生成 pw 序列（范围 [0, 200]），长度 N × 8
    pw_seq = np.random.uniform(0.0, 200.0, size=(N, 8)).astype(np.float32)

    # 3) 初始化各缓冲区
    pw_buf = deque([np.zeros(8, dtype=np.float32) for _ in range(W)], maxlen=W)
    imu_buf_A = deque([np.zeros(6, dtype=np.float32) for _ in range(W)], maxlen=W)  # 始终零
    imu_buf_B = deque([np.zeros(6, dtype=np.float32) for _ in range(W)], maxlen=W)  # 反馈策略

    preds_A = []  # 存放方案 A 的预测
    preds_B = []  # 存放方案 B 的预测

    # 4) 模型前向循环
    for step in range(N):
        # 每一步把当前 pw 加入 pw_buf
        pw_buf.append(pw_seq[step])

        # —— 方案 A：IMU 缓冲始终为零 —— #
        pw_tensor_A = torch.from_numpy(np.stack(pw_buf)[None, ...]).to(device)    # (1, W, 8)
        imu_tensor_A = torch.from_numpy(np.stack(imu_buf_A)[None, ...]).to(device)  # (1, W, 6)
        with torch.no_grad():
            pred_A = model(pw_tensor_A, imu_tensor_A)[0].cpu().numpy()  # (6,)
        preds_A.append(pred_A)

        # —— 方案 B：用反馈 IMU —— #
        pw_tensor_B = torch.from_numpy(np.stack(pw_buf)[None, ...]).to(device)
        imu_tensor_B = torch.from_numpy(np.stack(imu_buf_B)[None, ...]).to(device)
        with torch.no_grad():
            pred_B = model(pw_tensor_B, imu_tensor_B)[0].cpu().numpy()  # (6,)
        preds_B.append(pred_B)
        # 将本次预测作为下一步的 IMU 输入
        imu_buf_B.append(pred_B)

    # 转为 NumPy 数组以便后续处理
    preds_A = np.stack(preds_A)  # shape (N, 6)
    preds_B = np.stack(preds_B)

    # 5) 打印随机 pw（仅前 W 步，方便检查）
    print("[DEBUG] 前 %d 步随机 pw 序列：" % W)
    for t in range(W):
        print(f"  Step {t:02d}: pw = {pw_seq[t].tolist()}")
    print("")

    # 6) 统计并打印每个加速度维度在两种方案下的均值与标准差
    print("[SUMMARY] 各维度预测加速度的统计信息：")
    for i in range(6):
        A_mean, A_std = preds_A[:, i].mean(), preds_A[:, i].std()
        B_mean, B_std = preds_B[:, i].mean(), preds_B[:, i].std()
        print(
            f"  Dim {i:1d}: 方案 A -> mean={A_mean:.3f}, std={A_std:.3f}   |   "
            f"方案 B -> mean={B_mean:.3f}, std={B_std:.3f}"
        )
    print("")

    # 7) 绘制每个加速度维度随时间变化的对比图（各维度单独一个 figure）
    for i in range(6):
        plt.figure()
        plt.plot(range(N), preds_A[:, i], label="imu_zero", linestyle="solid")
        plt.plot(range(N), preds_B[:, i], label="imu_feedback", linestyle="dashed")
        plt.title(f"Predicted Acceleration Dimension {i}")
        plt.xlabel("Step")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
