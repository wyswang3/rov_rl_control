# training/target_generator.py
"""
模块化生成连续控制目标：位置轨迹和姿态轨迹，使用纯 numpy 实现，不依赖 scipy。
"""
import numpy as np
from typing import List, Tuple, Optional


def random_pose_target(
    position_range: Tuple[np.ndarray, np.ndarray] = (np.array([-1, -1, -1]), np.array([1, 1, 1])),
    quat_random: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    随机生成目标位置和四元数姿态。
    """
    low, high = position_range
    target_pos = np.random.uniform(low, high).astype(np.float32)
    if quat_random:
        u1, u2, u3 = np.random.rand(3)
        w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        target_quat = np.array([w, x, y, z], dtype=np.float32)
    else:
        target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return target_pos, target_quat


def random_path(
    waypoints: int = 5,
    bounds: Tuple[np.ndarray, np.ndarray] = (np.array([-1,-1,-1]), np.array([1,1,1]))
) -> List[np.ndarray]:
    """
    生成离散控制点列表，用于 path_following。
    """
    low, high = bounds
    return [np.random.uniform(low, high).astype(np.float32) for _ in range(waypoints)]


def linear_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    steps: int
) -> np.ndarray:
    """
    在 start 到 end 之间生成线性插值轨迹，返回 shape=(steps,3)。
    """
    return np.linspace(start, end, steps, dtype=np.float32)


def circular_trajectory(
    center: np.ndarray,
    radius: float,
    axis: np.ndarray,
    steps: int
) -> np.ndarray:
    """
    绕指定轴生成圆形轨迹，返回 shape=(steps,3)。
    """
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    # 构造平面基向量
    tmp = np.array([1,0,0], dtype=np.float32) if abs(axis[0])<0.9 else np.array([0,1,0], dtype=np.float32)
    v1 = np.cross(axis, tmp)
    v1 /= (np.linalg.norm(v1) + 1e-8)
    v2 = np.cross(axis, v1)
    angles = np.linspace(0, 2*np.pi, steps, endpoint=False)
    traj = np.stack([center + radius*(np.cos(a)*v1 + np.sin(a)*v2) for a in angles], axis=0).astype(np.float32)
    return traj


def random_piecewise_linear_trajectory(
    num_waypoints: int,
    bounds: Tuple[np.ndarray, np.ndarray],
    steps: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    在 bounds 范围内生成 num_waypoints 个随机点并用线性插值，返回 shape=(steps,3)。
    """
    if seed is not None:
        np.random.seed(seed)
    low, high = bounds
    ctrl_pts = np.random.uniform(low, high, size=(num_waypoints, 3)).astype(np.float32)
    xp = np.linspace(0, steps-1, num_waypoints)
    xi = np.arange(steps)
    traj = np.zeros((steps, 3), dtype=np.float32)
    for dim in range(3):
        traj[:, dim] = np.interp(xi, xp, ctrl_pts[:, dim])
    return traj


def orientation_along_path(
    positions: np.ndarray
) -> np.ndarray:
    """
    根据位置轨迹计算期望四元数，返回 shape=(steps,4)。
    """
    def vec_to_quat(forward: np.ndarray, up: np.ndarray = np.array([0,0,1],dtype=np.float32)):
        f = forward / (np.linalg.norm(forward)+1e-8)
        r = np.cross(up, f)
        r /= (np.linalg.norm(r)+1e-8)
        u = np.cross(f, r)
        R = np.stack([r, u, f], axis=1)
        qw = 0.5*np.sqrt(1+np.trace(R))
        qx = (R[2,1]-R[1,2])/(4*qw+1e-8)
        qy = (R[0,2]-R[2,0])/(4*qw+1e-8)
        qz = (R[1,0]-R[0,1])/(4*qw+1e-8)
        return np.array([qw, qx, qy, qz], dtype=np.float32)
    steps = positions.shape[0]
    quats = np.zeros((steps, 4), dtype=np.float32)
    for i in range(steps-1):
        quats[i] = vec_to_quat(positions[i+1] - positions[i])
    quats[-1] = quats[-2]
    return quats


def generate_trajectory(
    mode: str,
    steps: int,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成位置与姿态连续轨迹。
    mode: 'linear','circular','piecewise_linear'
    """
    if mode == 'linear':
        pos = linear_trajectory(kwargs['start'], kwargs['end'], steps)
    elif mode == 'circular':
        pos = circular_trajectory(kwargs['center'], kwargs['radius'], kwargs['axis'], steps)
    elif mode == 'piecewise_linear':
        pos = random_piecewise_linear_trajectory(
            kwargs['num_waypoints'], tuple(map(np.array, kwargs['bounds'])), steps, kwargs.get('seed')
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    ori = orientation_along_path(pos)
    return pos, ori
