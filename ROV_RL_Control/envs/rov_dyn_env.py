#!/usr/bin/env python3
# envs/rov_dyn_env.py

"""
Gym 环境：基于 HybridDynamicsModel 的加速度／位置／姿态控制任务。

Observation (37 维):
    [ pos(3), vel(3), quat(4), angv(3),
      target_pos(3), target_quat(4),
      imu_accel(6), jerk(3), last_action(8) ]

Action (8 维)：推力功率 ∈ [0, max_power]

Reward:
    r = - k_pos  * ||pos - target_pos||
        - k_att  * ||angle_dist(quat, target_quat)||
        - k_vel  * ||vel||
        - k_jerk * ||jerk||
        - k_acc  * ||accel_f - target_accel||
        - k_eng  * ∑(action^2)
        + k_succ * 1{pos_error < pos_tol}
"""

import gymnasium as gym
import numpy as np
import torch
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, Tuple

from utils.math_util import (
    quat_mul,
    normalize_vec,
    quat_from_omega,
    lowpass_filter,
    angle_dist,
    integrate_accel,
    integrate_velocity,
    integrate_ang_acc,
    quat_to_rot_matrix,
    body_to_nav_vel,
)
from models.lstm_dyn.loader import load_dynamics

# ─── 日志配置（只写 rov_env_log.txt，不打印到终端） ───────────────────────────
import logging
from pathlib import Path

LOG_FILE = Path.cwd() / "rov_env_log.txt"
logger = logging.getLogger("ROVDynEnv")
logger.setLevel(logging.INFO)

# 如果之前有任何 handler，先移除
for h in list(logger.handlers):
    logger.removeHandler(h)

# 只添加 FileHandler
fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
fh.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
fh.setFormatter(fmt)
logger.addHandler(fh)

# 重要：阻止消息传递到根 logger，否则会在终端打印
logger.propagate = False
# ─────────────────────────────────────────────────────────────────────────────


class ROVDynEnv(gym.Env):
    """
    ROV 动力学控制环境，使用 LSTM+HybridDynamicsModel 预测加速度，基于参考轨迹进行跟踪。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.02,
        max_power: float = 60.0,
        device: str = "cpu",
        window_size: int = 9,
        # —— 奖励权重 —— #
        k_pos: float = 0.0,
        k_att: float = 0.00,
        k_vel: float = 0.01,
        k_jerk: float = 0.01,
        k_acc: float = 0.1,
        k_eng: float = 0.1,
        k_succ: float = 5.0,
        pos_tol: float = 0.1,
        accel_filter_alpha: float = 0.5,
        traj_name: str = "hover",
    ):
        """
        初始化 ROV 控制环境

        参数:
          - dt: 时间步长 (s)
          - max_power: 最大推进器功率 (W)
          - device: "cpu" 或 "cuda"
          - window_size: LSTM 输入序列长度
          - k_pos, k_att, k_vel, k_jerk, k_acc, k_eng, k_succ, pos_tol: 奖励权重
          - accel_filter_alpha: 加速度低通滤波系数
          - traj_name: 参考轨迹文件名前缀，例如 "hover", "circle" 等
        """
        super().__init__()

        # —— 基本参数 —— #
        self.dt = dt
        self.max_power = max_power
        self.device = device
        self.window = window_size
        self.alpha = accel_filter_alpha

        # —— 奖励权重 —— #
        self.k_pos = k_pos
        self.k_att = k_att
        self.k_vel = k_vel
        self.k_jerk = k_jerk
        self.k_acc = k_acc
        self.k_eng = k_eng
        self.k_succ = k_succ
        self.pos_tol = pos_tol

        logger.info(
            f"[INIT] dt={dt}, max_power={max_power}, device={device}, window_size={window_size}, "
            f"accel_filter_alpha={accel_filter_alpha}\n"
            f"       k_pos={k_pos}, k_att={k_att}, k_vel={k_vel}, k_jerk={k_jerk}, "
            f"k_acc={k_acc}, k_eng={k_eng}, k_succ={k_succ}, pos_tol={pos_tol}\n"
            f"       traj_name={traj_name}"
        )

        # —— 定义动作空间与观测空间 —— #
        self.action_space = gym.spaces.Box(
            low=0.0, high=self.max_power, shape=(8,), dtype=np.float32
        )
        obs_high = np.full(37, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # —— 初始化内部状态 缓冲区与模型—— #
        self._init_state_buffers()
        self._init_dynamics_model()

        # —— 加载参考轨迹与加速度 —— #
        self._load_reference(traj_name)

        # —— 初始化环境变量 —— #
        self._reset_internal_state()

    def _init_state_buffers(self) -> None:
        """
        初始化用于动力学积分与滤波的缓冲区：
          - acc_buf: 最近 2 帧原始加速度
          - ang_acc_buf: 最近 2 帧角加速度（用于计算 jerk）
          - imu_buf: 最近 window 帧滤波后加速度
          - pw_buf: 最近 window 帧推力功率
          - last_actions: 存储上一帧动作
        """
        self.acc_buf = deque(maxlen=2)
        self.ang_acc_buf = deque(maxlen=2)
        self.imu_buf = deque(maxlen=self.window)
        self.pw_buf = deque(maxlen=self.window)
        self.last_actions = deque(maxlen=1)

    def _init_dynamics_model(self) -> None:
        """
        加载混合动力学模型 (HybridDynamicsModel)。模型用于根据历史推力和历史加速度预测加速度。
        """
        self.model, _ = load_dynamics(self.device)
        self.model.eval()
        logger.info("[MODEL] HybridDynamicsModel loaded and set to eval mode.")

    def _load_reference(self, traj_name: str) -> None:
        """
        加载 data/ref_trajs/{traj_name}_traj.npy 与 accel_{traj_name}.npy，
        并提取 pos、quat、accel 序列。
        """
        base_dir = Path(__file__).resolve().parent.parent / "data" / "ref_trajs"
        traj_path = base_dir / f"{traj_name}_traj.npy"
        accel_path = base_dir / f"accel_{traj_name}.npy"

        raw_traj = np.load(traj_path)  # (T, 13)
        self.ref_pos_traj = raw_traj[:, 0:3].astype(np.float32)   # (T, 3)
        self.ref_quat_traj = raw_traj[:, 3:7].astype(np.float32)  # (T, 4)
        self.ref_accel_traj = np.load(accel_path).astype(np.float32)  # (T, 6)
        self.max_steps = self.ref_pos_traj.shape[0]

        logger.info(f"[DATA] Loaded trajectory '{traj_name}' ({self.max_steps} steps)")

    def _reset_internal_state(self) -> None:
        """
        在 reset() 或初始化时，设置内部状态与缓冲区为零，以及设定目标为轨迹首帧。
        """
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.angv = np.zeros(3, dtype=np.float32)

        # 加速度与角加速度缓冲
        zero_acc = np.zeros(6, dtype=np.float32)
        self.acc_buf.clear()
        self.acc_buf.extend([zero_acc.copy(), zero_acc.copy()])

        zero_ang_acc = np.zeros(3, dtype=np.float32)
        self.ang_acc_buf.clear()
        self.ang_acc_buf.extend([zero_ang_acc.copy(), zero_ang_acc.copy()])

        # IMU 与推力缓冲
        zero_pw = np.zeros(8, dtype=np.float32)
        zero_imu = np.zeros(6, dtype=np.float32)
        self.pw_buf.clear()
        self.imu_buf.clear()
        for _ in range(self.window):
            self.pw_buf.append(zero_pw.copy())
            self.imu_buf.append(zero_imu.copy())

        # last_actions
        self.last_actions.clear()
        self.last_actions.append(zero_pw.copy())

        # 上一步 jerk（用于 observation）
        self.last_jerk = np.zeros(3, dtype=np.float32)

        # 步数计数器
        self.step_count = 0

        # 目标状态设为轨迹首帧
        self.target_pos = self.ref_pos_traj[0].copy()
        self.target_quat = self.ref_quat_traj[0].copy()
        self.target_accel = self.ref_accel_traj[0].copy()

    def reset(self, *, seed: int = None, options: Any = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境:
          - 随机初始化 pos、quat（随机 yaw）、vel、angv 置零
          - 清空并填充缓冲区
          - step_count 置 0
          - 目标设为参考轨迹第 0 帧

        返回:
          obs (37 维)、空 info dict
        """
        super().reset(seed=seed)
        rng = self.np_random

        # 随机初始化位置和姿态（随机 yaw）
        self.pos[:] = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        self.vel[:] = 0.0
        yaw0 = rng.uniform(0.0, 2 * np.pi)
        half = 0.5 * yaw0
        s = np.sin(half)
        self.quat[:] = np.array([0.0, 0.0, s, np.cos(half)], dtype=np.float32)
        self.angv[:] = 0.0

        # 清空并初始化缓冲
        self._reset_internal_state()

        logger.info(
            f"[RESET] pos={self.pos.tolist()}, quat={self.quat.tolist()}, "
            f"target_pos={self.target_pos.tolist()}, target_quat={self.target_quat.tolist()}"
        )
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步更新:
          1) Clip 动作，并存入 pw_buf, last_actions
          2) LSTM 模型预测原始加速度
          3) 去除重力分量 (z 方向减 9.81)
          4) 低通滤波得到滤波后加速度 accel_f
          5) 计算角加速度差分 jerk, 更新 ang_acc_buf, last_jerk
          6) 线加速度欧拉积分 → 更新 vel, pos (并 clamp)
          7) 角加速度欧拉积分 → 更新 angv, quat (并 clamp)
          8) 计算 reward 与 info
          9) 更新 imu_buf, step_count, 目标状态到下一帧
          10) 每 500 步写一条日志（预测加速度、速度、位置、目标、各项误差、reward、control_input）
          11) 返回 (obs, reward, done=False, truncated=False, info)
        """
        # (1) Clip & 记录推力动作
        act = np.clip(action, 0.0, self.max_power).astype(np.float32)
        self.pw_buf.append(act)
        self.last_actions.append(act.copy())

        # (2) LSTM 推理预测原始加速度
        accel_pred = self._predict_accel()

        # (3) 去除重力分量（假设 z 方向需减去 9.81）
        accel_pred[2] -= 9.81

        # (4) 加速度低通滤波
        accel_f = self._filter_accel(accel_pred)

        # (5) 计算角加速度差分 (jerk) 并更新缓冲
        curr_ang_acc = accel_f[3:].astype(np.float32)
        prev_ang_acc = self.ang_acc_buf[-1]
        self.ang_acc_buf.append(curr_ang_acc.copy())
        jerk = (curr_ang_acc - prev_ang_acc).astype(np.float32)
        self.last_jerk = jerk.copy()

        # (6) 线加速度欧拉积分 → 更新 vel (m/s), pos (m)
        lin_acc = accel_f[:3]
        new_vel = (self.vel + lin_acc * self.dt).astype(np.float32)
        # clamp 线速度到 [-3, +3] m/s
        self.vel = np.clip(new_vel, -3.0, 3.0).astype(np.float32)
        self.pos = (self.pos + self.vel * self.dt).astype(np.float32)

        # (7) 角加速度欧拉积分 → 更新 angv (rad/s), quat
        ang_acc = accel_f[3:]
        new_angv = (self.angv + ang_acc * self.dt).astype(np.float32)
        # clamp 角速度到 [-ω_max, +ω_max]，ω_max = 50°/s ≈ 0.87266 rad/s
        omega_max = 50.0 * np.pi / 180.0
        self.angv = np.clip(new_angv, -omega_max, omega_max).astype(np.float32)
        dq = quat_from_omega(self.angv, self.dt)
        quat_updated = quat_mul(self.quat, dq)
        self.quat = normalize_vec(quat_updated).astype(np.float32)

        # (8) 计算 reward 与 info
        reward, info = self._compute_reward(act, accel_f, jerk)

        # (9) 更新 IMU 缓冲、步数计数、目标状态
        self.imu_buf.append(accel_f.copy())
        self.step_count += 1
        idx = min(self.step_count, self.max_steps - 1)
        self.target_pos = self.ref_pos_traj[idx].copy()
        self.target_quat = self.ref_quat_traj[idx].copy()
        self.target_accel = self.ref_accel_traj[idx].copy()

        # (10) 每 500 步写一条详细日志
        if self.step_count % 500 == 0:
            pos_err = np.linalg.norm(self.pos - self.target_pos)
            att_err = angle_dist(self.quat, self.target_quat)
            vel_norm = np.linalg.norm(self.vel)
            acc_err = np.linalg.norm(accel_f - self.target_accel)
            eng_pen = np.dot(act, act)
            jerk_norm = np.linalg.norm(jerk)

            logger.info(
                f"[STEP {self.step_count}] "
                f"action={act.tolist()}, accel_pred={accel_pred.tolist()}, accel_f={accel_f.tolist()},\n"
                f"           vel={self.vel.tolist()} (||vel||={vel_norm:.2f}), pos={self.pos.tolist()} (err={pos_err:.2f}),\n"
                f"           quat={self.quat.tolist()}, target_pos={self.target_pos.tolist()}, target_quat={self.target_quat.tolist()} (att_err={att_err:.2f}),\n"
                f"           acc_err={acc_err:.2f}, eng_pen={eng_pen:.2f}, jerk_pen={jerk_norm:.2f}, reward={reward:.2f}"
            )

        # (11) 返回 (obs, reward, done, truncated, info)
        done = False
        truncated = False
        return self._get_obs(), float(reward), done, truncated, info

    def _predict_accel(self) -> np.ndarray:
        """
        使用 LSTM 模型 (HybridDynamicsModel) 根据最近 window 帧的 pw_buf 和 imu_buf
        预测当前原始加速度 (6 维：前三为线加速度，后三为角加速度)，返回 shape (6,) np.float32。
        """
        pw_seq = torch.tensor(
            np.stack(self.pw_buf)[None, ...],  # shape (1, window, 8)
            dtype=torch.float32,
            device=self.device,
        )
        imu_seq = torch.tensor(
            np.stack(self.imu_buf)[None, ...],  # shape (1, window, 6)
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            accel_t = self.model(pw_seq, imu_seq)  # shape (1, 6)
        return accel_t[0].cpu().numpy().astype(np.float32)

    def _filter_accel(self, raw_accel: np.ndarray) -> np.ndarray:
        """
        对原始加速度 raw_accel (6,) 做低通滤波，返回滤波后 accel_f (6,)。
        """
        self.acc_buf.append(raw_accel)
        seq = np.vstack(self.acc_buf)  # shape (2, 6)
        filt_seq = lowpass_filter(seq, self.alpha)  # shape (2, 6)
        return filt_seq[-1].astype(np.float32)

    def _compute_reward(
        self, action: np.ndarray, accel_f: np.ndarray, jerk: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        计算本步的 reward，并返回 (reward, info)。
        info 包含位姿、速度、加速度、jerk、目标、动作，以及各类误差变量。
        """
        pos_err = float(np.linalg.norm(self.pos - self.target_pos))
        att_err = angle_dist(self.quat, self.target_quat)
        vel_norm = float(np.linalg.norm(self.vel))
        jerk_norm = float(np.linalg.norm(jerk))
        eng_pen = float(np.dot(action, action))
        acc_err = float(np.linalg.norm(accel_f - self.target_accel))
        succ_bonus = float(self.k_succ if pos_err < self.pos_tol else 0.0)

        reward = (
            - self.k_pos * pos_err
            - self.k_att * att_err
            - self.k_vel * vel_norm
            - self.k_jerk * jerk_norm
            - self.k_acc * acc_err
            - self.k_eng * eng_pen
            + succ_bonus
        )

        # 简要日志（每 100 步）
        if self.step_count % 100 == 0:
            logger.info(
                f"[STEP {self.step_count}] "
                f"pos_err={pos_err:.2f}, att_err={att_err:.2f}, vel={vel_norm:.2f}, "
                f"jerk_pen={self.k_jerk * jerk_norm:.2f}, acc_pen={self.k_acc * acc_err:.2f}, "
                f"eng_pen={self.k_eng * eng_pen:.2f} -> reward={reward:.2f}"
            )

        info: Dict[str, Any] = {
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "quat": self.quat.copy(),
            "angv": self.angv.copy(),
            "accel": accel_f.copy(),
            "jerk": jerk.copy(),
            "target_pos": self.target_pos.copy(),
            "target_quat": self.target_quat.copy(),
            "action": action.copy(),
            "pos_error": pos_err,
            "att_error": att_err,
            "acc_error": acc_err,
        }
        return reward, info

    def _get_obs(self) -> np.ndarray:
        """
        构造 37 维观测向量：
          [ pos(3), vel(3), quat(4), angv(3),
            target_pos(3), target_quat(4),
            imu_accel(6), jerk(3), last_action(8) ]
        """
        return np.concatenate(
            [
                self.pos,                  # 3
                self.vel,                  # 3
                self.quat,                 # 4
                self.angv,                 # 3
                self.target_pos,           # 3
                self.target_quat,          # 4
                self.imu_buf[-1],          # 6
                self.last_jerk,            # 3
                self.last_actions[-1],     # 8
            ],
            axis=0,
        ).astype(np.float32)
