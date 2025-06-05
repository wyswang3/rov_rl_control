"""
utils/math_util.py  —  纯 NumPy 实现的常用数学工具
"""

import numpy as np

__all__ = [
    "normalize_vec",
    "quat_mul",
    "quat_from_omega",
    "angle_dist",
    "quat_to_rot_matrix",
    "body_to_nav_vel",
    "lowpass_filter",
    "integrate_accel",
    "integrate_velocity",
    "integrate_ang_acc",
]


def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    将向量 v 按 L2 范数归一化，返回新数组。
    如果 ||v|| 很小，则直接返回原向量以避免数值不稳定。
    """
    norm = np.linalg.norm(v)
    if norm < eps:
        return v.copy().astype(np.float32)
    return (v / norm).astype(np.float32)


def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    四元数乘法 (Hamilton product)，q、r 均按 [x, y, z, w] 排列。
    返回新的四元数 [x, y, z, w]。
    """
    x0, y0, z0, w0 = q
    x1, y1, z1, w1 = r
    return np.array([
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    ], dtype=np.float32)


def quat_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    """
    根据角速度 omega (rad/s) 和时间步长 dt，计算对应的四元数增量 [x, y, z, w]。
    公式：θ = ||ω|| * dt， axis = ω / ||ω||，四元数 = [axis * sin(θ/2), cos(θ/2)]。
    如果 ||ω||*dt 很小，则返回单位四元数 [0, 0, 0, 1]。
    """
    theta = np.linalg.norm(omega) * dt
    if theta < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = omega / (np.linalg.norm(omega) + 1e-12)
    half = 0.5 * theta
    s = np.sin(half)
    return np.hstack([axis * s, np.cos(half)]).astype(np.float32)


def angle_dist(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    计算两个四元数 q1、q2（按 [x, y, z, w] 排列）之间的夹角距离。
    假设 q1, q2 都已归一化。公式：angle = arccos(2 * (q·q)^2 - 1)。
    返回标量弧度值。
    """
    dot = float(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return float(np.arccos(2 * dot * dot - 1))


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """
    将四元数 q 转换为旋转矩阵 R (3×3)，假设 q 按 [x, y, z, w] 排列并已归一化。
    返回的 R 可用于将机体坐标系速度转换到导航坐标系：
        v_nav = R @ v_body
    """
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx + zz),       2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx),   1 - 2*(xx + yy)],
    ], dtype=np.float32)


def body_to_nav_vel(q: np.ndarray, vel_body: np.ndarray) -> np.ndarray:
    """
    将机体坐标系下的速度 vel_body (shape=(3,)) 转换到导航坐标系速度。
    q: 四元数 [x, y, z, w] 表示机体到导航系的旋转。
    返回 v_nav = R(q) @ vel_body。
    """
    R = quat_to_rot_matrix(q)
    return (R @ vel_body).astype(np.float32)


def lowpass_filter(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    一阶递归低通滤波（IIR）：
    data: 形状 (N, D)，表示 N 帧、D 维信号
    alpha: 滤波系数，在 (0,1] 范围。值越小，输出越平滑；越接近 1，响应越迅速。
    返回 shape 与 data 相同的滤波后结果。
    公式：
      y[0] = data[0]
      y[t] = alpha * data[t] + (1 - alpha) * y[t-1]
    """
    N, D = data.shape
    y = np.zeros((N, D), dtype=np.float32)
    y[0] = data[0]
    for t in range(1, N):
        y[t] = alpha * data[t] + (1.0 - alpha) * y[t - 1]
    return y


def integrate_accel(acc_seq: np.ndarray, dt: float) -> np.ndarray:
    """
    将线性加速度序列 acc_seq (shape=(N,3)) 欧拉积分得到速度序列。
    公式：
      vel[0] = acc_seq[0] * dt
      vel[t] = vel[t-1] + acc_seq[t] * dt
    返回 shape=(N, 3) 的速度序列，其中 vel[t] 对应时刻 t 的速度。
    """
    N, _ = acc_seq.shape
    vel = np.zeros((N, 3), dtype=np.float32)
    vel[0] = acc_seq[0] * dt
    for t in range(1, N):
        vel[t] = vel[t - 1] + acc_seq[t] * dt
    return vel


def integrate_velocity(vel_seq: np.ndarray, dt: float) -> np.ndarray:
    """
    将线速度序列 vel_seq (shape=(N,3)) 欧拉积分得到位置序列。
    公式：
      pos[0] = vel_seq[0] * dt
      pos[t] = pos[t-1] + vel_seq[t] * dt
    返回 shape=(N, 3) 的位置序列，其中 pos[t] 对应时刻 t 的位置。
    """
    N, _ = vel_seq.shape
    pos = np.zeros((N, 3), dtype=np.float32)
    pos[0] = vel_seq[0] * dt
    for t in range(1, N):
        pos[t] = pos[t - 1] + vel_seq[t] * dt
    return pos


def integrate_ang_acc(ang_acc_seq: np.ndarray, dt: float) -> np.ndarray:
    """
    将角加速度序列 ang_acc_seq (shape=(N,3)) 欧拉积分得到角速度序列。
    公式：
      ang_vel[0] = ang_acc_seq[0] * dt
      ang_vel[t] = ang_vel[t-1] + ang_acc_seq[t] * dt
    返回 shape=(N, 3) 的角速度序列，其中 ang_vel[t] 对应时刻 t 的角速度。
    """
    N, _ = ang_acc_seq.shape
    ang_vel = np.zeros((N, 3), dtype=np.float32)
    ang_vel[0] = ang_acc_seq[0] * dt
    for t in range(1, N):
        ang_vel[t] = ang_vel[t - 1] + ang_acc_seq[t] * dt
    return ang_vel
