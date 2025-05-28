"""
utils.math_util — 纯 NumPy 实现的常用数学工具
"""
import numpy as np

__all__ = [
    "normalize_vec",
    "quat_mul",
    "quat_from_omega",
]

def normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按 L2 范数归一化向量，返回新数组。"""
    norm = np.linalg.norm(v) + eps
    return v / norm

def quat_mul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """四元数乘法 (Hamilton product)，q、r 均按 [x, y, z, w] 排列。"""
    x0, y0, z0, w0 = q
    x1, y1, z1, w1 = r
    return np.array([
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    ], dtype=np.float32)

def quat_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    """角速度 (rad/s) 积分 dt 秒，得到增量四元数 [x, y, z, w]。"""
    theta = np.linalg.norm(omega) * dt
    if theta < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    axis = omega / (np.linalg.norm(omega) + 1e-12)
    half = 0.5 * theta
    s = np.sin(half)
    return np.hstack([axis * s, np.cos(half)]).astype(np.float32)
