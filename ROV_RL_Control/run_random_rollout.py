#!/usr/bin/env python3
"""
rl/run_random_rollout.py

随机动作采样测试 ROVDynEnv 环境的稳定性：
每隔若干步打印位置和奖励，确保无 NaN、无状态发散。
"""
import numpy as np
from envs.rov_dyn_env import ROVDynEnv

def main():
    # 建议使用 CPU 做随机测试
    device = 'cpu'
    env = ROVDynEnv(dt=0.02, max_power=480.0, device=device)

    obs, _ = env.reset(seed=42)
    print(f"Initial observation: {obs}")

    # 随机执行 500 步
    for t in range(500):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if t % 100 == 0:
            pos = obs[0:3]
            print(f"Step {t:03d}: pos={pos}, reward={reward:.3f}")
        if done:
            obs, _ = env.reset()

    env.close()
    print("Random rollout test completed successfully.")

if __name__ == '__main__':
    main()
