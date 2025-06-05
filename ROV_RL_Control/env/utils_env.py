# 文件：rov_rl_control/env/utils_env.py

import numpy as np
from typing import Any, Dict, Tuple


def quaternion_to_euler(quat: np.ndarray) -> Tuple[float, float, float]:
    """
    将四元数 [qx, qy, qz, qw] 转换为欧拉角 (roll, pitch, yaw)，ZYX 顺序：
      roll  (φ)  = atan2(2(qw*qx + qy*qz), 1 − 2(qx² + qy²))
      pitch (θ)  = asin (2(qw*qy − qz*qx))
      yaw   (ψ)  = atan2(2(qw*qz + qx*qy), 1 − 2(qy² + qz²))

    参数：
      - quat: np.ndarray, shape=(4,), [qx, qy, qz, qw]

    返回：
      - (roll, pitch, yaw), 单位：弧度
    """
    qx, qy, qz, qw = quat
    # roll (φ)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (θ)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2)  # Gimbal lock
    else:
        pitch = np.arcsin(sinp)

    # yaw (ψ)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    四元数乘法 q = q1 ⊗ q2，假设四元数格式均为 [qx, qy, qz, qw]。

    参数：
      - q1, q2: np.ndarray, shape=(4,), [qx, qy, qz, qw]

    返回：
      - q: np.ndarray, shape=(4,), [qx, qy, qz, qw]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([x, y, z, w], dtype=np.float32)


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """
    将四元数归一化为单位四元数。

    参数：
      - quat: np.ndarray, shape=(4,)

    返回：
      - 单位四元数 np.ndarray, shape=(4,)
    """
    norm = np.linalg.norm(quat) + 1e-8
    return (quat / norm).astype(np.float32)


def rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
    """
    根据四元数 [qx, qy, qz, qw] 计算 3×3 旋转矩阵，顺序 ZYX（yaw-pitch-roll）。

    参数：
      - quat: np.ndarray, shape=(4,), [qx, qy, qz, qw]

    返回：
      - R: np.ndarray, shape=(3,3)
    """
    qx, qy, qz, qw = quat
    # 预计算
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R = np.array([
        [1.0 - 2.0 * (yy + zz),       2.0 * (xy - wz),          2.0 * (xz + wy)],
        [      2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz),         2.0 * (yz - wx)],
        [      2.0 * (xz - wy),       2.0 * (yz + wx),    1.0 - 2.0 * (xx + yy)]
    ], dtype=np.float32)

    return R


def compute_attitude_error(
    quat: np.ndarray, target_quat: np.ndarray
) -> float:
    """
    计算当前姿态 quat 与目标姿态 target_quat 之间的角度误差：
      error = 2 * arccos( |<quat, target_quat>| )

    参数：
      - quat: np.ndarray, shape=(4,)
      - target_quat: np.ndarray, shape=(4,)

    返回：
      - ang_err: float, 单位：弧度
    """
    dot = np.dot(quat, target_quat)
    dot_clipped = np.clip(abs(dot), -1.0, 1.0)
    ang_err = 2.0 * np.arccos(dot_clipped)
    return float(ang_err)


def random_target_position(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float]
) -> np.ndarray:
    """
    随机生成一个目标位置，均匀分布在给定范围内。

    参数：
      - x_range: (xmin, xmax)
      - y_range: (ymin, ymax)
      - z_range: (zmin, zmax)

    返回：
      - target_pos: np.ndarray, shape=(3,)
    """
    x = np.random.uniform(x_range[0], x_range[1])
    y = np.random.uniform(y_range[0], y_range[1])
    z = np.random.uniform(z_range[0], z_range[1])
    return np.array([x, y, z], dtype=np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两点 a, b 之间的欧式距离。

    参数：
      - a, b: np.ndarray, shape=(3,)

    返回：
      - dist: float
    """
    return float(np.linalg.norm(a - b))


def clamp_state(
    state: np.ndarray,
    pos_limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    vel_limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
) -> np.ndarray:
    """
    对状态（位置和速度）做上下限裁剪，以保证数值稳定。

    参数：
      - state: np.ndarray, shape=(13,), [pos(3), quat(4), lin_vel(3), ang_vel(3)]
      - pos_limits: ((xmin, xmax), (ymin, ymax), (zmin, zmax))
      - vel_limits: ((uxmin, uxmax), (uymin, uymax), (uzmin, uzmax))

    返回：
      - clipped_state: np.ndarray, shape=(13,)
    """
    clipped = state.copy()
    # 裁剪位置
    clipped[0]  = np.clip(clipped[0], pos_limits[0][0], pos_limits[0][1])
    clipped[1]  = np.clip(clipped[1], pos_limits[1][0], pos_limits[1][1])
    clipped[2]  = np.clip(clipped[2], pos_limits[2][0], pos_limits[2][1])
    # 裁剪线速度
    clipped[7]  = np.clip(clipped[7], vel_limits[0][0], vel_limits[0][1])
    clipped[8]  = np.clip(clipped[8], vel_limits[1][0], vel_limits[1][1])
    clipped[9]  = np.clip(clipped[9], vel_limits[2][0], vel_limits[2][1])
    # 角速度可视需求裁剪
    # clipped[10] = ...
    # clipped[11] = ...
    # clipped[12] = ...
    return clipped


def state_to_obs(
    state: np.ndarray,
    target_pose: Dict[str, np.ndarray],
    prev_action: np.ndarray
) -> np.ndarray:
    """
    从完整状态构造观测（与 ROVEnv._get_observation 相同逻辑），
    便于外部调用时也能快速获得相同观测格式。

    参数：
      - state: np.ndarray, shape=(13,), [pos(3), quat(4), lin_vel(3), ang_vel(3)]
      - target_pose: {"position": np.ndarray(3,), "quat": np.ndarray(4,)}
      - prev_action: np.ndarray, shape=(8,)

    返回：
      - obs: np.ndarray, shape=(25,)
    """
    pos = state[0:3]
    quat = state[3:7]
    lin_vel = state[7:10]
    ang_vel = state[10:13]
    depth = np.array([pos[2]], dtype=np.float32)
    pos_error = target_pose["position"] - pos
    obs = np.concatenate([
        pos,
        quat,
        lin_vel,
        ang_vel,
        depth,
        pos_error,
        prev_action
    ], axis=0).astype(np.float32)
    return obs
