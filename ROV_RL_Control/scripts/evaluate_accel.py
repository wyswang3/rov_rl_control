#!/usr/bin/env python3
"""
scripts/evaluate_accel.py

离线评估加速度跟踪精度：
  1) 加载训练好的 PPO 策略
  2) 对指定参考加速度轨迹进行多次 rollout
  3) 计算每步 ||aₜ–a*ₜ|| 误差，绘制均值±标准差曲线
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from envs.rov_dyn_env import ROVDynEnv

def evaluate(model_path: str,
             ref_traj_path: str,
             num_episodes: int = 5,
             dt: float = 0.02):
    # 加载参考加速度轨迹
    ref = np.load(ref_traj_path)  # shape (steps,6)
    steps = ref.shape[0]

    # 加载模型
    model = PPO.load(model_path, device="cpu")

    all_errs = []
    for ep in range(num_episodes):
        env = ROVDynEnv(dt=dt, device="cpu")
        obs, _ = env.reset()
        errs = []
        for t in range(steps):
            # 策略选择动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            real_acc = info.get("accel", env.last_acc)  # 确保能拿到
            target_acc = ref[t]
            errs.append(np.linalg.norm(real_acc - target_acc))
            if done:
                break
        all_errs.append(errs)
        env.close()

    all_errs = np.array(all_errs)  # (num_episodes, steps)
    mean_err = all_errs.mean(axis=0)
    std_err  = all_errs.std(axis=0)

    # 绘图
    t = np.arange(steps) * dt
    plt.figure(figsize=(8,4))
    plt.fill_between(t, mean_err - std_err, mean_err + std_err,
                     color="C0", alpha=0.3, label="±1 std")
    plt.plot(t, mean_err, "C0", label="mean error")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration error (m/s²)")
    plt.title("Acceleration Tracking Error")
    plt.legend()
    out_png = Path("eval_acc_error.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved acceleration error plot → {out_png}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="checkpoints/ppo_latest.zip",
                   help="PPO 模型路径")
    p.add_argument("-r", "--ref",   default="data/ref_trajs/accel_ref_traj.npy",
                   help="参考加速度轨迹文件")
    p.add_argument("-n", "--num",   type=int, default=5,
                   help="评估 rollout 次数")
    args = p.parse_args()
    evaluate(args.model, args.ref, num_episodes=args.num)
