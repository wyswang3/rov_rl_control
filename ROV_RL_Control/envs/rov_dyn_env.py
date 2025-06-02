#!/usr/bin/env python3
# envs/rov_dyn_env.py

"""
Gym 环境：基于 HybridDynamicsModel 的加速度/位置/姿态控制任务

- Observation (36 维)：
    [ pos(3), vel(3), quat(4), angv(3),
      target_pos(3), target_quat(4),
      imu_accel(6), jerk(3),
      last_action(8) ]

- Action (8 维)：推力功率 ∈ [0, max_power]

- Reward（示例，仅供参考）：
    r = - k_pos  * ||pos - target_pos||        # 位置误差惩罚
        - k_att  * att_error                   # 姿态误差惩罚
        - k_vel  * ||vel||                     # 速度惩罚（抑制振荡）
        - k_jerk * ||jerk||                    # 角加速度变化惩罚（运动平滑）
        - k_eng  * ∑(action^2)                  # 能耗惩罚
        + k_succ * (pos_error < pos_tol)        # 若到达目标位置则加奖励
"""

import gymnasium as gym
import numpy as np
import torch
from collections import deque

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
        # —— 加权系数 —— #
        k_pos: float = 1.0,      # 位置误差惩罚
        k_att: float = 0.5,      # 姿态误差惩罚
        k_vel: float = 0.1,      # 速度惩罚
        k_jerk: float = 0.01,    # 角加速度变化惩罚
        k_eng: float = 0.001,    # 能耗惩罚
        k_succ: float = 5.0,     # 成功到达目标的奖励
        pos_tol: float = 0.1,    # 目标位置容忍距离 (m)
        accel_filter_alpha: float = 0.5,
    ):
        """
        构造函数参数（由 make_vec_env 传入）：
          - dt, max_power, device, window_size: 同之前含义
          - k_pos, k_att, k_vel, k_jerk, k_eng, k_succ, pos_tol: 奖励函数系数
          - accel_filter_alpha: 加速度低通滤波系数
        """
        super().__init__()

        # —— 基本参数 —— #
        self.dt         = dt
        self.max_power  = max_power
        self.device     = device
        self.window     = window_size
        self.alpha      = accel_filter_alpha

        # —— 奖励系数 —— #
        self.k_pos   = k_pos
        self.k_att   = k_att
        self.k_vel   = k_vel
        self.k_jerk  = k_jerk
        self.k_eng   = k_eng
        self.k_succ  = k_succ
        self.pos_tol = pos_tol  # 到达目标距离阈值

        # —— 加载混合动力学模型 (HybridDynamicsModel) —— #
        # 返回 (model, init_hidden_fn)，这里只取 model
        self.model, _ = load_dynamics(self.device)
        self.model.eval()

        # —— 定义动作空间与观测空间 —— #
        # 动作：8 维推力功率，范围 [0, max_power]
        self.action_space = gym.spaces.Box(
            low=0.0, high=self.max_power, shape=(8,), dtype=np.float32
        )

        # 观测维度 36 = 3(pos) +3(vel) +4(quat) +3(angv)
        #              +3(target_pos)+4(target_quat)
        #              +6(imu滤波后加速度)+3(jerk)
        #              +8(last_action)
        obs_high = np.full(36, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # —— 系统状态变量 —— #
        # 位置 pos (3), 速度 vel (3)
        # 姿态用四元数 quat (4), 角速度 angv (3)
        self.pos   = np.zeros(3, dtype=np.float32)
        self.vel   = np.zeros(3, dtype=np.float32)
        # 初始单位四元数：无旋转
        self.quat  = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.angv  = np.zeros(3, dtype=np.float32)

        # —— 目标状态 —— #
        # target_pos (3), target_quat (4)
        self.target_pos  = np.zeros(3, dtype=np.float32)
        # 目标也用四元数表示
        self.target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # —— 历史缓冲区 —— #
        # 原始加速度缓冲 acc_buf，用于低通滤波输入 (maxlen=2)
        # 每个元素为 6 维：[lin_acc(3), ang_acc(3)]
        self.acc_buf  = deque(maxlen=2)
        # jerk_buf: 上一帧的角加速度变化(3 维)
        self.jerk_buf = deque(maxlen=1)
        # pw_buf: 记录前 window_size 帧的 8 维推力功率
        self.pw_buf   = deque(maxlen=window_size)
        # imu_buf:  保存前 window_size 帧滤波后加速度 (6 维)
        self.imu_buf  = deque(maxlen=window_size)
        # last_actions: 保存最近一帧动作 (8 维)
        self.last_actions = deque(maxlen=1)

    def reset(self, *, seed=None, options=None):
        """
        重置环境，返回 obs, {}。
        1) 随机初始化 pos、quat（只随机 yaw）
        2) 其余物理量置零
        3) 清空并填充各缓冲区
        """
        super().reset(seed=seed)
        rng = self.np_random

        # —— 随机初始化位置、姿态（只随机 yaw） —— #
        self.pos[:]  = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        self.vel[:]  = 0.0
        # 初始姿态：roll=pitch=0，yaw 随机
        yaw0 = rng.uniform(0.0, 2 * np.pi)
        # 将三轴欧拉转换为四元数（小角度近似）
        # 先把欧拉[0,0,yaw0]→四元量
        # 四元数公式：axis = [0,0,1], angle=yaw0
        half = 0.5 * yaw0
        s = np.sin(half)
        self.quat = np.array([0.0, 0.0, s, np.cos(half)], dtype=np.float32)
        self.angv[:] = 0.0

        # —— 清空并初始化加速度/jerk 缓冲 —— #
        zero_acc   = np.zeros(6, dtype=np.float32)  # [lin_acc(3), ang_acc(3)]
        self.acc_buf.clear()
        self.acc_buf.append(zero_acc.copy())
        self.acc_buf.append(zero_acc.copy())

        zero_jerk  = np.zeros(3, dtype=np.float32)
        self.jerk_buf.clear()
        self.jerk_buf.append(zero_jerk.copy())

        # —— 清空并初始化推力、滤波后加速度缓冲 —— #
        zero_pw    = np.zeros(8, dtype=np.float32)
        zero_imu   = np.zeros(6, dtype=np.float32)
        self.pw_buf.clear()
        self.imu_buf.clear()
        for _ in range(self.window):
            self.pw_buf.append(zero_pw.copy())
            self.imu_buf.append(zero_imu.copy())

        # —— 清空并初始化 last_actions —— #
        self.last_actions.clear()
        self.last_actions.append(zero_pw.copy())

        # —— 初始化目标—— #
        # 这里示例直接把目标位置/姿态都设为原点处“悬停”，
        # 你可以在 reset 之后，根据需求改写 self.target_pos 与 self.target_quat
        self.target_pos[:]  = np.zeros(3, dtype=np.float32)
        self.target_quat[:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # 返回初始观测
        return self._get_obs(), {}

    def step(self, action):
        """
        执行一步：
         1) Clip & 记录当前动作：act ∈ [0, max_power] ∈ R⁸
         2) 构造 model 输入序列 (pw_seq, imu_seq)
         3) 调用混合动力学模型预测加速度 (6 维)
         4) 去除重力项（Z 轴向下为正，所以 accel[2] -= 9.81）
         5) 原始加速度入队，做低通滤波 (6 维) 得到 accel_f
         6) 计算角加速度差分 jerk (3 维)
         7) 线性加速度积分 → 更新 vel (3) 与 pos (3)
         8) 角加速度积分 → 更新 angv (3), 并用四元数 + 角速度更新 quat (4)
         9) 计算奖励 r，根据位置误差、姿态误差、速度、jerk、能耗等
        返回：obs (36 维), reward(float), done=False, truncated=False, info
        """
        # —— (1) Clip & 记录 last_actions, pw_buf —— #
        act = np.clip(action, 0.0, self.max_power).astype(np.float32)
        self.pw_buf.append(act)
        self.last_actions.append(act.copy())

        # —— (2) 构造模型输入：pw_seq (1, window, 8), imu_seq (1, window, 6) —— #
        pw_seq  = torch.tensor(
            np.stack(self.pw_buf)[None, ...],
            dtype=torch.float32,
            device=self.device
        )  # 形状: (1, window_size, 8)

        imu_seq = torch.tensor(
            np.stack(self.imu_buf)[None, ...],
            dtype=torch.float32,
            device=self.device
        )  # 形状: (1, window_size, 6)

        # —— (3) 模型推理：预测当前加速度 (1,6) —— #
        with torch.no_grad():
            accel_t = self.model(pw_seq, imu_seq)  # Tensor shape (1, 6)
        accel = accel_t[0].cpu().numpy().astype(np.float32)  # (6,)

        # —— (4) 去除重力 (Z 轴向下为正) —— #
        accel[2] -= 9.81

        # —— (5) 低通滤波 —— #
        self.acc_buf.append(accel)
        raw_seq  = np.vstack(self.acc_buf)  # 形状 (2, 6)
        filt_seq = lowpass_filter(raw_seq, self.alpha)
        # 取最后一行做为“滤波后加速度”
        accel_f = filt_seq[-1].astype(np.float32)  # (6,)

        # —— (6) 计算 jerk，仅针对角加速度分量(3~5) —— #
        jerk = accel_f[3:] - filt_seq[-2, 3:]
        jerk = jerk.astype(np.float32)
        self.jerk_buf.append(jerk.copy())

        # —— (7) 线性加速度积分 → 更新 vel, pos —— #
        #   integrate_accel 会对  filt_seq[:, :3] 做积分，返回 shape=(2,3)
        vel_seq = integrate_accel(filt_seq[:, :3], self.dt)
        self.vel = vel_seq[-1].astype(np.float32)

        #   再把速度序列积分得到位置序列 (在第一行填 [0,0,0])
        pos_seq = integrate_velocity(
            np.vstack([np.zeros(3, dtype=np.float32), vel_seq]),
            self.dt
        )
        self.pos = pos_seq[-1].astype(np.float32)

        # —— (8) 角加速度积分 → 更新 angv, quat —— #
        # 累加角加速度得到新角速度
        self.angv += accel_f[3:] * self.dt  # (3,)
        # 用四元数 dq 更新姿态 quat
        # 先把当前 quat 视作上一时刻四元数
        dq = quat_from_omega(self.angv, self.dt)   # (4,)
        new_quat = quat_mul(self.quat, dq)         # 四元数乘法
        # 归一化
        new_quat = normalize_vec(new_quat)
        self.quat = new_quat.astype(np.float32)

        # —— (9) 计算奖励 —— #
        # 1) 位置误差
        pos_error = np.linalg.norm(self.pos - self.target_pos).astype(np.float32)

        # 2) 姿态误差：用四元数内积计算角度差
        #    att_error = arccos(2*(q·q_target)^2 - 1)
        dot_q = float(np.dot(self.quat, self.target_quat))
        dot_q = np.clip(dot_q, -1.0, 1.0)
        att_error = np.arccos(2 * dot_q**2 - 1).astype(np.float32)

        # 3) 速度惩罚
        vel_norm = float(np.linalg.norm(self.vel))

        # 4) jerk 惩罚
        jerk_norm = float(np.linalg.norm(jerk))

        # 5) 能耗惩罚
        eng_pen = float(np.dot(act, act))

        # 6) 成功到达目标奖励 (若位置误差小于 pos_tol)
        success_bonus = float(self.k_succ if pos_error < self.pos_tol else 0.0)

        # 汇总
        reward = (
            - self.k_pos  * pos_error
            - self.k_att  * att_error
            - self.k_vel  * vel_norm
            - self.k_jerk * jerk_norm
            - self.k_eng  * eng_pen
            + success_bonus
        )

        # —— (10) 更新 imu_buf：后续 next step 输入使用 filt_seq 的最后一行 —— #
        self.imu_buf.append(accel_f.copy())

        # —— info 字典返回当前关键信息 —— #
        info = {
            "pos":   self.pos.copy(),       # 当前 3D 位置
            "vel":   self.vel.copy(),       # 当前 3D 线速度
            "quat":  self.quat.copy(),      # 当前四元数姿态
            "angv":  self.angv.copy(),      # 当前角速度 (3)
            "accel": accel_f.copy(),        # 当前滤波后加速度 (6)
            "jerk":  jerk.copy(),           # 当前角加加速度差分 (3)
            "target_pos": self.target_pos.copy(),
            "target_quat": self.target_quat.copy(),
            "action": act.copy(),           # 本轮所用动作 (8)
            "pos_error": pos_error,         # 位置误差 (scalar)
            "att_error": att_error,         # 姿态误差 (scalar)
        }

        done = False  # 本示例不提前结束
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
