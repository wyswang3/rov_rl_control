#!/usr/bin/env python3
# envs/rov_dyn_env.py

"""
Gym 环境：基于 HybridDynamicsModel 的加速度／位置／姿态控制任务

- Observation (36 维)：
    [ pos(3), vel(3), quat(4), angv(3),
      target_pos(3), target_quat(4),
      imu_accel(6), jerk(3),
      last_action(8) ]

- Action (8 维)：推力功率 ∈ [0, max_power]

- Reward：
    r = - k_pos  * ||pos - target_pos||        # 位置误差惩罚
        - k_att  * ||angle_dist(quat, target_quat)|| # 姿态误差惩罚
        - k_vel  * ||vel||                     # 速度惩罚（抑制振荡）
        - k_jerk * ||jerk||                    # 角加速度变化惩罚（运动平滑）
        - k_acc  * ||accel_f - target_accel||  # 加速度跟踪误差惩罚
        - k_eng  * ∑(action^2)                  # 能耗惩罚
        + k_succ * (pos_error < pos_tol)        # 成功到达目标位置奖励
"""

import gymnasium as gym
import numpy as np
import torch
from collections import deque
from pathlib import Path

from utils.math_util import (
    quat_mul,
    lowpass_filter,
    integrate_accel,
    integrate_velocity,
    normalize_vec,
    quat_from_omega
)
from models.lstm_dyn.loader import load_dynamics


class ROVDynEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.02,
        max_power: float = 60.0,
        device: str = "cpu",
        window_size: int = 9,
        # —— 奖励系数 —— #
        k_pos: float = 1.0,        # 位置误差惩罚系数
        k_att: float = 0.5,        # 姿态误差惩罚系数
        k_vel: float = 0.1,        # 速度惩罚系数
        k_jerk: float = 0.01,      # 角加速度变化惩罚系数
        k_acc: float = 1.0,        # 加速度跟踪误差惩罚系数
        k_eng: float = 0.001,      # 能耗惩罚系数
        k_succ: float = 5.0,       # 成功到达目标奖励
        pos_tol: float = 0.1,      # 目标位置容忍距离 (m)
        accel_filter_alpha: float = 0.5,
        traj_name: str = "hover",  # 指定要加载哪条参考轨迹（对应 data/ref_trajs/{traj_name}_traj.npy / accel_{traj_name}.npy）
    ):
        """
        参数说明：
          - dt, max_power, device, window_size: 同之前含义
          - k_pos, k_att, k_vel, k_jerk, k_acc, k_eng, k_succ, pos_tol: 奖励权重
          - accel_filter_alpha: 加速度低通滤波系数
          - traj_name: 参考轨迹的文件名前缀，比如 "hover"、"circle" 等
        """
        super().__init__()

        # —— 基本参数 —— #
        self.dt         = dt
        self.max_power  = max_power
        self.device     = device
        self.window     = window_size
        self.alpha      = accel_filter_alpha

        # —— 奖励权重 —— #
        self.k_pos   = k_pos
        self.k_att   = k_att
        self.k_vel   = k_vel
        self.k_jerk  = k_jerk
        self.k_acc   = k_acc
        self.k_eng   = k_eng
        self.k_succ  = k_succ
        self.pos_tol = pos_tol

        # —— 加载混合动力学模型 (HybridDynamicsModel) —— #
        self.model, _ = load_dynamics(self.device)
        self.model.eval()

        # —— 动作空间 & 观测空间 —— #
        self.action_space = gym.spaces.Box(
            low=0.0, high=self.max_power, shape=(8,), dtype=np.float32
        )
        # 观测：36 维
        # [pos(3), vel(3), quat(4), angv(3),
        #  target_pos(3), target_quat(4),
        #  imu_accel(6), jerk(3),
        #  last_action(8)]
        obs_high = np.full(36, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # —— 环境内部状态 —— #
        self.pos   = np.zeros(3, dtype=np.float32)    # 位置
        self.vel   = np.zeros(3, dtype=np.float32)    # 线速度
        self.quat  = np.array([0, 0, 0, 1], dtype=np.float32)  # 单位四元数
        self.angv  = np.zeros(3, dtype=np.float32)    # 角速度

        # —— 目标状态 —— #
        self.target_pos    = np.zeros(3, dtype=np.float32)
        self.target_quat   = np.array([0, 0, 0, 1], dtype=np.float32)
        # 当前要跟踪的“真实”加速度
        self.target_accel  = None  # 会在 reset() 里赋值为形状 (T,6) 的数组

        # —— 历史缓冲区 —— #
        self.acc_buf       = deque(maxlen=2)   # 原始加速度缓冲 (用来做低通滤波)
        self.jerk_buf      = deque(maxlen=1)   # 仅存储上一帧的角加加速度差分
        self.pw_buf        = deque(maxlen=window_size)  # 最近 window_size 帧的推力功率
        self.imu_buf       = deque(maxlen=window_size)  # 最近 window_size 帧的“滤波后”加速度
        self.last_actions  = deque(maxlen=1)            # 仅存储上一帧动作

        # —— 用于跟踪当前时刻索引 —— #
        self.step_count    = 0

        # —— 参考数据路径 & 加载 —— #
        data_root = Path(__file__).resolve().parent.parent / "data" / "ref_trajs"
        traj_path  = data_root / f"{traj_name}_traj.npy"
        accel_path = data_root / f"accel_{traj_name}.npy"

        # 1) 加载轨迹文件 (假设格式：每行 3(pos) + 4(quat) + 6(额外信息可忽略))
        #    这里我们只关心 pos(0:3) 和 quat(3:7)，如果轨迹 npy 里多存了姿态
        raw_traj = np.load(traj_path)  # e.g. shape (T, 13)
        # 将前 3 列视作位置，3:7 列视作四元数
        self.ref_pos_traj   = raw_traj[:, 0:3].astype(np.float32)  # shape (T,3)
        self.ref_quat_traj  = raw_traj[:, 3:7].astype(np.float32)  # shape (T,4)

        # 2) 加载参考加速度 (shape (T,6))：前 3 是线加速度，后 3 是角加速度
        self.ref_accel_traj = np.load(accel_path).astype(np.float32)  # shape (T,6)

        # 3) 记录轨迹长度
        self.max_steps = self.ref_pos_traj.shape[0]


    def reset(self, *, seed=None, options=None):
        """
        重置环境：
          - pos/quart/yaw 随机初始化
          - 其余状态置零
          - step_count 置 0
          - 各缓冲区置零并填充初始值
          - target_* 直接赋为参考轨迹的第 0 帧
        返回 (obs, {})，obs 维度 (36,)
        """
        super().reset(seed=seed)
        rng = self.np_random

        # —— 初始化位置/速度/姿态/角 speed —— #
        self.pos[:]   = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        self.vel[:]   = 0.0
        yaw0 = rng.uniform(0.0, 2*np.pi)
        half = 0.5 * yaw0
        s = np.sin(half)
        self.quat[:] = np.array([0.0, 0.0, s, np.cos(half)], dtype=np.float32)
        self.angv[:] = 0.0

        # —— 清空并初始化加速度和 jerk 缓冲 —— #
        zero_acc = np.zeros(6, dtype=np.float32)
        self.acc_buf.clear()
        self.acc_buf.append(zero_acc.copy())
        self.acc_buf.append(zero_acc.copy())

        zero_jerk = np.zeros(3, dtype=np.float32)
        self.jerk_buf.clear()
        self.jerk_buf.append(zero_jerk.copy())

        # —— 清空并初始化推力与滤波后加速度缓冲 —— #
        zero_pw  = np.zeros(8, dtype=np.float32)
        zero_imu = np.zeros(6, dtype=np.float32)
        self.pw_buf.clear()
        self.imu_buf.clear()
        for _ in range(self.window):
            self.pw_buf.append(zero_pw.copy())
            self.imu_buf.append(zero_imu.copy())

        # —— 清空并初始化 last_actions —— #
        self.last_actions.clear()
        self.last_actions.append(zero_pw.copy())

        # —— 计时器归零 —— #
        self.step_count = 0

        # —— 设定当前目标为第 0 帧 —— #
        self.target_pos   = self.ref_pos_traj[0].copy()
        self.target_quat  = self.ref_quat_traj[0].copy()
        self.target_accel = self.ref_accel_traj[0].copy()

        # —— 返回初始观测 —— #
        return self._get_obs(), {}


    def step(self, action):
        """
        执行一步：
          1) Clip & 把动作记入 pw_buf、last_actions
          2) 用最近 window 帧的 pw_buf/imu_buf 做 LSTM 输入，预测当前加速度
          3) 去除重力 (可选，由模型本身决定)
          4) 原始加速度入队，用 lowpass_filter 得到 accel_f (6 维)
          5) 计算 jerk = accel_f[3:] - 上一帧 filt_seq[-2,3:]
          6) 将线性加速度积分 -> 更新 vel, pos
          7) 将角加速度积分 -> 更新 angv, quat
          8) 奖励 = - k_pos * ||pos - target_pos|| 
                    - k_att * att_error 
                    - k_vel * ||vel||
                    - k_jerk * ||jerk||
                    - k_acc  * ||accel_f - target_accel||
                    - k_eng  * ||action||^2
                  + k_succ if pos_error < pos_tol
          9) 更新 imu_buf, step_count, 并更新 target_* 到下一帧
        返回 obs (36 维)、reward、done=False、False、info
        """
        # —(1) Clip & 记录动作—#
        act = np.clip(action, 0.0, self.max_power).astype(np.float32)
        self.pw_buf.append(act)
        self.last_actions.append(act.copy())

        # —(2) LSTM 推理，预测原始加速度—#
        pw_seq  = torch.tensor(
            np.stack(self.pw_buf)[None, ...],  # (1, window, 8)
            dtype=torch.float32, device=self.device
        )
        imu_seq = torch.tensor(
            np.stack(self.imu_buf)[None, ...],  # (1, window, 6)
            dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            accel_t = self.model(pw_seq, imu_seq)  # (1,6)
        accel = accel_t[0].cpu().numpy().astype(np.float32)  # (6,)

        # —(3) 如有需要可手动去除重力，例如 accel[2] -= 9.81 ——#
        #    本例中假设模型输出已剔除重力，或者不需要显式剔除

        # —(4) 低通滤波——#
        self.acc_buf.append(accel)
        raw_seq  = np.vstack(self.acc_buf)        # (2,6)
        filt_seq = lowpass_filter(raw_seq, self.alpha)  # (2,6)
        accel_f  = filt_seq[-1].astype(np.float32)      # (6,)

        # —(5) 计算 jerk——#
        jerk = accel_f[3:] - filt_seq[-2, 3:]  # 3 维
        jerk = jerk.astype(np.float32)
        self.jerk_buf.append(jerk.copy())

        # —(6) 线加速度积分 → 更新 vel, pos ——#
        vel_seq = integrate_accel(filt_seq[:, :3], self.dt)  # shape (2,3)
        self.vel = vel_seq[-1].astype(np.float32)
        pos_seq = integrate_velocity(
            np.vstack([np.zeros(3, dtype=np.float32), vel_seq]), self.dt
        )  # shape (2,3)
        self.pos = pos_seq[-1].astype(np.float32)

        # —(7) 角加速度积分 → 更新 angv, quat ——#
        self.angv += accel_f[3:] * self.dt  # (3,)
        dq = quat_from_omega(self.angv, self.dt)  # (4,) 四元数增量
        new_quat = quat_mul(self.quat, dq)        # 四元数乘法
        new_quat = normalize_vec(new_quat)        # 归一化
        self.quat = new_quat.astype(np.float32)

        # —(8) 计算各类误差——#
        # 位置误差
        pos_error = float(np.linalg.norm(self.pos - self.target_pos))
        # 姿态误差（四元数）: att_error = arccos(2*(q·q_target)^2 - 1)
        dot_q = float(np.dot(self.quat, self.target_quat))
        dot_q = np.clip(dot_q, -1.0, 1.0)
        att_error = float(np.arccos(2 * dot_q**2 - 1))

        # 速度惩罚
        vel_norm = float(np.linalg.norm(self.vel))
        # jerk 惩罚
        jerk_norm = float(np.linalg.norm(jerk))
        # 能耗惩罚
        eng_pen = float(np.dot(act, act))

        # 加速度跟踪误差：||accel_f - target_accel||
        acc_error = float(np.linalg.norm(accel_f - self.target_accel))

        # 成功到达奖励（仅末状态判定）
        success_bonus = float(self.k_succ if pos_error < self.pos_tol else 0.0)

        # 汇总 reward
        reward = (
            - self.k_pos   * pos_error
            - self.k_att   * att_error
            - self.k_vel   * vel_norm
            - self.k_jerk  * jerk_norm
            - self.k_acc   * acc_error
            - self.k_eng   * eng_pen
            + success_bonus
        )

        # —(9) 更新历史缓冲 & 步数 & 下一帧目标——#
        self.imu_buf.append(accel_f.copy())
        self.step_count += 1

        # 如果尚未到达参考轨迹末尾，就把 target_* 更新到下一帧，否则保持最后一帧
        idx = min(self.step_count, self.max_steps - 1)
        self.target_pos   = self.ref_pos_traj[idx].copy()
        self.target_quat  = self.ref_quat_traj[idx].copy()
        self.target_accel = self.ref_accel_traj[idx].copy()

        # 返回 obs, reward, done, truncated, info
        info = {
            "pos":         self.pos.copy(),
            "vel":         self.vel.copy(),
            "quat":        self.quat.copy(),
            "angv":        self.angv.copy(),
            "accel":       accel_f.copy(),
            "jerk":        jerk.copy(),
            "target_pos":  self.target_pos.copy(),
            "target_quat": self.target_quat.copy(),
            "action":      act.copy(),
            "pos_error":   pos_error,
            "att_error":   att_error,
            "acc_error":   acc_error,
        }
        done = False  # 此处不提前结束
        return self._get_obs(), float(reward), done, False, info

    def _get_obs(self):
        """
        构造 36 维观测： 
          [ pos(3), vel(3),
            quat(4), angv(3),
            target_pos(3), target_quat(4),
            imu_accel(6), jerk(3),
            last_action(8) ]
        """
        return np.concatenate([
            self.pos,                       # 3
            self.vel,                       # 3
            self.quat,                      # 4
            self.angv,                      # 3
            self.target_pos,                # 3
            self.target_quat,               # 4
            self.imu_buf[-1],               # 6 (滤波后加速度)
            self.jerk_buf[-1],              # 3 (角加速度差分)
            self.last_actions[-1]           # 8 (上一帧动作)
        ], axis=0).astype(np.float32)
