#!/usr/bin/env python3
# envs/rov_dyn_env.py

"""
Gym 环境：基于 HybridDynamicsModel 的加速度控制任务
Observation (15-dim):
  [eul(3), ang_vel(3), last_acc(6), last_jerk(3)]
Action (8-dim): 推力功率
Reward:
  - w_err * ||accel||^2
  - w_jerk * ||jerk||^2
  - w_eng * sum(power^2)
"""

import gymnasium as gym
import numpy as np
import torch
from collections import deque
from utils.math_util import quat_mul, quat_from_omega
from models.lstm_dyn.loader import load_dynamics

class ROVDynEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 dt: float = 0.02,
                 max_power: float = 80.0,
                 device: str = "cpu",
                 window_size: int = 9,
                 w_err: float = 1.0,
                 w_jerk: float = 0.5,
                 w_eng: float = 0.01):
        super().__init__()
        self.dt = dt
        self.max_power = max_power
        self.device = device
        self.window = window_size

        # Loss weights
        self.w_err = w_err
        self.w_jerk = w_jerk
        self.w_eng = w_eng

        # Load HybridDynamicsModel
        self.model, _ = load_dynamics(device)
        self.model.eval()

        # Action and observation spaces
        self.action_space = gym.spaces.Box(0.0, max_power, (8,), np.float32)
        # Observations: [eul(3), ang_vel(3), last_acc(6), last_jerk(3)] = 15-dim
        obs_high = np.inf * np.ones(15, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # State buffers
        self.eul_buf   = deque(maxlen=1)  # store last euler
        self.angv_buf  = deque(maxlen=1)  # store last angular velocity
        self.acc_buf   = deque(maxlen=1)  # last acceleration
        self.jerk_buf  = deque(maxlen=1)  # last jerk

        # Initialize zero-history
        zero_eul = np.zeros(3, dtype=np.float32)
        zero_ang = np.zeros(3, dtype=np.float32)
        zero_acc = np.zeros(6, dtype=np.float32)
        zero_jerk= np.zeros(3, dtype=np.float32)
        self.eul_buf.append(zero_eul)
        self.angv_buf.append(zero_ang)
        self.acc_buf.append(zero_acc)
        self.jerk_buf.append(zero_jerk)

        # Internal LSTM history
        self.pw_buf  = deque(maxlen=window_size)
        self.imu_buf = deque(maxlen=window_size)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Reset orientation and angular velocity
        init_eul = np.array([0.0, 0.0, rng.uniform(0, 2*np.pi)], dtype=np.float32)
        init_ang = np.zeros(3, dtype=np.float32)
        zero_acc = np.zeros(6, dtype=np.float32)
        zero_jrk = np.zeros(3, dtype=np.float32)

        self.eul_buf.clear(); self.eul_buf.append(init_eul)
        self.angv_buf.clear(); self.angv_buf.append(init_ang)
        self.acc_buf.clear(); self.acc_buf.append(zero_acc)
        self.jerk_buf.clear(); self.jerk_buf.append(zero_jrk)

        # Reset LSTM buffers
        zero_pw  = np.zeros(8, dtype=np.float32)
        zero_imu = np.zeros(6, dtype=np.float32)
        self.pw_buf.clear(); self.imu_buf.clear()
        for _ in range(self.window):
            self.pw_buf.append(zero_pw)
            self.imu_buf.append(zero_imu)

        return self._get_obs(), {}

    def step(self, action):
        """
        执行动作一步：
        - action: 推力功率 (8,)
        - 返回: (obs, reward, done, False, {})
        """
        # 1) 限幅 & 更新推力历史
        act = np.clip(action, 0.0, self.max_power).astype(np.float32)
        self.pw_buf.append(act)

        # 2) 构造 LSTM 输入张量 (1, window, *)
        pw_seq = torch.tensor(np.stack(self.pw_buf)[None], dtype=torch.float32, device=self.device)
        imu_seq = torch.tensor(np.stack(self.imu_buf)[None], dtype=torch.float32, device=self.device)

        # 3) 模型推理：预测加速度 (1,6) → (6,)
        with torch.no_grad():
            accel_t = self.model(pw_seq, imu_seq)
        accel = accel_t[0].cpu().numpy().astype(np.float32)

        # 4) 重力去除（假设 Z 轴指向下）
        accel[2] -= 9.81

        # 5) 计算 jerk（角加速度部分）
        prev_acc = self.acc_buf[-1]
        jerk = accel[3:] - prev_acc[3:]

        # 6) 计算奖励：加速度误差、jerk、能耗
        r_err = -self.w_err * np.dot(accel, accel)
        r_jerk = -self.w_jerk * np.dot(jerk, jerk)
        r_eng = -self.w_eng * np.dot(act, act)
        reward = float(r_err + r_jerk + r_eng)

        # 7) 更新历史缓冲
        self.imu_buf.append(accel)
        self.acc_buf.append(accel)
        self.jerk_buf.append(jerk)

        # 8) 环境不终止
        done = False

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        eul   = self.eul_buf[-1]
        angv  = self.angv_buf[-1]
        acc   = self.acc_buf[-1]
        jerk  = self.jerk_buf[-1]
        return np.concatenate([eul, angv, acc, jerk]).astype(np.float32)
