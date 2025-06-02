"""
utils/math_util.py  —  纯 NumPy 实现的常用数学工具
"""

import numpy as np

__all__ = [
    "normalize_vec",
    "quat_mul",
    "quat_from_omega",
    "lowpass_filter",
    "integrate_accel",
    "integrate_velocity",
]


def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    按 L2 范数归一化向量，返回新数组。
    """
    norm = np.linalg.norm(v) + eps
    return v / norm


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
    根据角速度 (rad/s) 和时间步长 dt，计算对应的四元数增量 [x, y, z, w]。
    公式：θ = ||ω|| * dt， axis = ω / ||ω||，四元数 = [axis * sin(θ/2), cos(θ/2)]。
    如果 ||ω||*dt 很小，则返回单位四元数 [0,0,0,1]。
    """
    # omega: 形状 (3,)
    theta = np.linalg.norm(omega) * dt
    if theta < 1e-8:
        # 角度几乎为零，返回单位四元数
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    axis = omega / (np.linalg.norm(omega) + 1e-12)
    half = 0.5 * theta
    s = np.sin(half)
    return np.hstack([axis * s, np.cos(half)]).astype(np.float32)


def lowpass_filter(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    一阶递归低通滤波（IIR）：
    data: 形状 (N, D)，分别为 N 帧、D 维信号
    alpha: 滤波系数，在 (0,1] 范围。数值越小，更新越平滑；越接近 1，响应越迅速。
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
    把线性加速度序列 acc_seq (shape=(N,3)) 积分得到速度序列。
    简单欧拉积分：
      v[0] = 0
      v[t] = v[t-1] + acc_seq[t-1] * dt
    返回 shape=(N, 3) 的速度序列，其中最后一行对应最新速度。
    """
    N, _ = acc_seq.shape
    vel = np.zeros((N, 3), dtype=np.float32)
    for t in range(1, N):
        vel[t] = vel[t - 1] + acc_seq[t - 1] * dt
    return vel


def integrate_velocity(vel_seq: np.ndarray, dt: float) -> np.ndarray:
    """
    把线速度序列 vel_seq (shape=(N,3)) 积分得到位置序列。
    简单欧拉积分：
      p[0] = 0
      p[t] = p[t-1] + vel_seq[t-1] * dt
    返回 shape=(N, 3) 的位置序列，其中最后一行对应最新位置。
    """
    N, _ = vel_seq.shape
    pos = np.zeros((N, 3), dtype=np.float32)
    for t in range(1, N):
        pos[t] = pos[t - 1] + vel_seq[t - 1] * dt
    return pos
