#!/usr/bin/env python3
"""
scripts/generate_data_configs.py

生成物理可行的 ROV 运动参考数据，包括：
  1. 域随机化配置：data/domain_rand.yaml
  2. 轨迹与对应加速度文件：data/ref_trajs/{name}_traj.npy, data/ref_trajs/accel_{name}.npy

当前支持的轨迹类型：
  - hover      : 悬停
  - point_to_point: 点对点直线运动，保持水平姿态
  - circle     : XY 平面匀速圆周
  - z_wave     : Z 方向正弦摆动
  - inspection : 简单巡视任务，围绕目标点做小幅姿态变化
"""

import os
import yaml
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from pathlib import Path
from typing import Dict, List, Tuple


# ———— 全局常量 ———— #
DT    = 0.02             # 时间步长 (s)
T     = 20.0             # 总时长 (s)
STEPS = int(T / DT)      # 时间步数
TIME  = np.linspace(0, T, STEPS, endpoint=False)

# 域随机化范围
DOMAIN_RAND = {
    "mass_scale":      [0.9, 1.1],
    "drag_scale":      [0.8, 1.2],
    "accel_noise_std": 0.02,
    "gyro_noise_std":  0.01,
}

# 输出路径
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REF_DIR  = DATA_DIR / "ref_trajs"
DATA_DIR.mkdir(exist_ok=True)
REF_DIR.mkdir(exist_ok=True)


# ———— 四元数工具 ———— #
def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    球面线性插值 (SLERP) between quaternion q1 and q2, parameter t ∈ [0,1].
    四元数格式: [w, x, y, z]
    """
    # 将 [w, x, y, z] → [x, y, z, w] 供 scipy Rotation 使用
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
    # 使用 Slerp 类
    slerp_func = Slerp([0.0, 1.0], Rotation.concatenate([r1, r2]))
    r_interp = slerp_func([t])[0]  # 传入一个列表，用 [0.0,1.0] 为关键时刻
    x, y, z, w = r_interp.as_quat()
    return np.array([w, x, y, z], dtype=np.float32)


def average_quaternions(quats: np.ndarray) -> np.ndarray:
    """
    计算一组四元数的加权平均（通过特征向量法），返回归一化后的平均四元数 [w, x, y, z]。
    quats: shape (N,4), 每行 [w, x, y, z]
    """
    Q = quats.copy().astype(np.float64)
    # 构造对称矩阵 M = Q^T Q
    M = Q.T @ Q
    eigenvals, eigenvecs = np.linalg.eigh(M)
    avg = eigenvecs[:, -1]
    return (avg / np.linalg.norm(avg)).astype(np.float32)


# ———— 路径／速度剖面生成 ———— #
def linear_path(start: np.ndarray, end: np.ndarray, steps: int = STEPS) -> np.ndarray:
    """
    生成两点之间的等间距直线路径，返回 shape (steps, 3)。
    start, end 均为长度 3 的数组。
    """
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    if steps <= 1:
        return np.tile(start, (1, 1))
    path = np.linspace(start, end, steps, axis=0)
    return path.astype(np.float32)


def trapezoidal_velocity_profile(path: np.ndarray,
                                 max_speed: float = 0.5,
                                 max_accel: float = 0.3,
                                 dt: float = DT) -> np.ndarray:
    """
    对给定路径（shape: (steps,3)）生成梯形速度剖面，返回速度向量序列 (steps,3)。
    - 总路径长度 = 累积各段 ||Δpos|| 之和
    - 若路径过短，则退化为三角形剖面
    """
    steps = path.shape[0]
    if steps < 2:
        return np.zeros((steps, 3), dtype=np.float32)

    # 计算各段距离
    deltas = path[1:] - path[:-1]
    segment_dist = np.linalg.norm(deltas, axis=1)  # shape (steps-1,)
    total_dist = np.sum(segment_dist)

    # 计算加速/减速距离
    accel_time = max_speed / max_accel
    accel_dist = 0.5 * max_accel * (accel_time**2)

    velocities = np.zeros((steps, 3), dtype=np.float32)
    if total_dist < 1e-6:
        return velocities  # 整体不动

    # 用于存累计距离
    cum_dist = np.concatenate([[0.0], np.cumsum(segment_dist)])

    def compute_speed(dist_along: float) -> float:
        """
        给定当前位置沿路径的累积距离 dist_along ∈ [0,total_dist]，
        返回当前位置的标量速度。
        """
        if total_dist < 2 * accel_dist:
            # 三角形剖面
            peak = np.sqrt(max_accel * total_dist)
            half = total_dist / 2
            if dist_along <= half:
                return peak * (dist_along / half)
            else:
                return peak * (1 - (dist_along - half) / half)
        else:
            # 梯形剖面
            if dist_along <= accel_dist:
                return max_accel * np.sqrt(2 * dist_along / max_accel)
            elif dist_along <= (total_dist - accel_dist):
                return max_speed
            else:
                # 减速阶段
                return max_speed * (1 - (dist_along - (total_dist - accel_dist)) / accel_dist)

    # 逐点计算速度大小，再沿当前切线方向恢复为速度向量
    for i in range(steps):
        dist_i = cum_dist[i]
        speed_scalar = compute_speed(dist_i)
        if i == 0:
            velocities[i] = np.zeros(3)
        else:
            if segment_dist[i - 1] < 1e-8:
                velocities[i] = np.zeros(3)
            else:
                direction = deltas[i - 1] / (segment_dist[i - 1] + 1e-12)
                velocities[i] = speed_scalar * direction
    return velocities


# ———— 轨迹生成主函数 ———— #
def generate_feasible_trajectory(target: Dict) -> List[Dict]:
    """
    根据 target 字典的字段，选择对应轨迹生成函数。
    返回 List[Dict]，每个 Dict 包含:
      {
        "pos": shape(3,),
        "quat": shape(4,),
        "vel": shape(3,),
        "ang_vel": shape(3,)
      }
    """
    ttype = target.get("type", "hover")
    if ttype == "hover":
        return gen_hover_trajectory(target)
    elif ttype == "point_to_point":
        return gen_point_to_point_trajectory(target)
    elif ttype == "circle":
        return gen_circle_trajectory(target)
    elif ttype == "z_wave":
        return gen_z_wave_trajectory(target)
    elif ttype == "inspection":
        return gen_inspection_trajectory(target)
    else:
        # 默认悬停
        return gen_hover_trajectory(target)


def gen_hover_trajectory(target: Dict) -> List[Dict]:
    """
    悬停轨迹：只生成常零位置与姿态。
    target 可包含 "pos"（3,）和 "quat"（4,），
    若无则默认 pos=[0,0,0], quat=[1,0,0,0]。
    """
    pos0 = np.asarray(target.get("pos", [0, 0, 0]), dtype=np.float32)
    quat0 = np.asarray(target.get("quat", [1, 0, 0, 0]), dtype=np.float32)
    trajectory = []
    for _ in range(STEPS):
        trajectory.append({
            "pos": pos0.copy(),
            "quat": quat0.copy(),
            "vel": np.zeros(3, dtype=np.float32),
            "ang_vel": np.zeros(3, dtype=np.float32),
        })
    return trajectory


def gen_circle_trajectory(target: Dict) -> List[Dict]:
    """
    圆周轨迹：XY 平面半径为 target["radius"] 或 1.0，Z=const。
    姿态 yaw 线性从 0 旋转至 2π，保持 roll=pitch=0。
    """
    radius = float(target.get("radius", 1.0))
    z0 = float(target.get("pos_z", 0.0))
    omega = 2 * np.pi / T

    trajectory = []
    for i, t in enumerate(TIME):
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        pos = np.array([x, y, z0], dtype=np.float32)
        yaw = (omega * t) % (2 * np.pi)
        quat = Rotation.from_euler("zyx", [yaw, 0.0, 0.0]).as_quat()
        # scipy 输出 [x,y,z,w] → 转 [w,x,y,z]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)
        trajectory.append({
            "pos": pos,
            "quat": quat,
            "vel": np.zeros(3, dtype=np.float32),
            "ang_vel": np.zeros(3, dtype=np.float32),
        })
    # 根据生成的位置，用梯形剖面算速度
    path = np.vstack([state["pos"] for state in trajectory])
    vels = trapezoidal_velocity_profile(path)
    for i in range(STEPS):
        trajectory[i]["vel"] = vels[i]
    return trajectory


def gen_z_wave_trajectory(target: Dict) -> List[Dict]:
    """
    正弦摆动：X=Y=常数（target["pos_xy"]），Z = amplitude * sin(2π t / T) + offset。
    姿态保持恒定 (quat=[1,0,0,0])。
    """
    amp = float(target.get("amplitude", 0.5))
    offset_z = float(target.get("offset_z", 0.0))
    pos_xy = np.asarray(target.get("pos_xy", [0.0, 0.0]), dtype=np.float32)

    trajectory = []
    for i, t in enumerate(TIME):
        z = offset_z + amp * np.sin(2 * np.pi * t / T)
        pos = np.array([pos_xy[0], pos_xy[1], z], dtype=np.float32)
        quat = np.array(target.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        trajectory.append({
            "pos": pos,
            "quat": quat,
            "vel": np.zeros(3, dtype=np.float32),
            "ang_vel": np.zeros(3, dtype=np.float32),
        })
    # 速度剖面
    path = np.vstack([s["pos"] for s in trajectory])
    vels = trapezoidal_velocity_profile(path)
    for i in range(STEPS):
        trajectory[i]["vel"] = vels[i]
    return trajectory


def gen_point_to_point_trajectory(target: Dict) -> List[Dict]:
    """
    点对点直线运动轨迹：从 target["start"] 到 target["end"]，
    姿态保持水平（quat=[1,0,0,0] 或 target["quat"]）。
    速度剖面使用梯形方案。
    """
    start = np.asarray(target["start"], dtype=np.float32)
    end   = np.asarray(target["end"], dtype=np.float32)
    path  = linear_path(start, end)
    vels  = trapezoidal_velocity_profile(path)

    quat0 = np.asarray(target.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
    trajectory = []
    for i in range(STEPS):
        trajectory.append({
            "pos": path[i],
            "quat": quat0.copy(),
            "vel": vels[i],
            "ang_vel": np.zeros(3, dtype=np.float32),
        })
    return trajectory


def gen_inspection_trajectory(target: Dict) -> List[Dict]:
    """
    简单巡视任务：在 target["pos"] 附近保持位置不变，
    依次在 target["quat_sequence"] 中插值小幅姿态（slerp）后保持。
    quat_sequence: List[np.ndarray] 每项均为 [4] 四元数
    """
    pos0 = np.asarray(target.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
    quats = [np.asarray(q, dtype=np.float32) for q in target["quat_sequence"]]
    n_phase = len(quats)
    phase_steps = STEPS // n_phase

    trajectory = []
    for idx in range(n_phase):
        q_start = quats[idx]
        q_end   = quats[(idx + 1) % n_phase]
        for k in range(phase_steps):
            alpha = k / max(phase_steps - 1, 1)
            q_interp = quaternion_slerp(q_start, q_end, alpha)
            trajectory.append({
                "pos": pos0.copy(),
                "quat": q_interp,
                "vel": np.zeros(3, dtype=np.float32),
                "ang_vel": np.zeros(3, dtype=np.float32),
            })

    # 如果长度不足 STEPS，则补最后一帧
    while len(trajectory) < STEPS:
        trajectory.append({
            "pos": pos0.copy(),
            "quat": quats[-1].copy(),
            "vel": np.zeros(3, dtype=np.float32),
            "ang_vel": np.zeros(3, dtype=np.float32),
        })
    return trajectory


# ———— 加速度／角加速度计算 ———— #
def compute_physical_acceleration(trajectory: List[Dict], dt: float = DT) -> np.ndarray:
    """
    给定 trajectory (List of Dict)，每项含 "vel"(3,) 和 "ang_vel"(3,),
    计算线加速度与角加速度，返回 shape (steps, 6):
      [lin_acc_x, lin_acc_y, lin_acc_z, ang_acc_x, ang_acc_y, ang_acc_z]
    """
    steps = len(trajectory)
    vel_seq = np.vstack([s["vel"] for s in trajectory])        # (steps,3)
    angvel_seq = np.vstack([s["ang_vel"] for s in trajectory])  # (steps,3)

    # 线加速度
    lin_acc = np.gradient(vel_seq, dt, axis=0)    # (steps,3)
    # 角加速度
    ang_acc = np.gradient(angvel_seq, dt, axis=0) # (steps,3)

    accel = np.hstack([lin_acc.astype(np.float32),
                       ang_acc.astype(np.float32)])
    return accel


# ———— 保存函数 ———— #
def save_trajectory(name: str, trajectory: List[Dict], accel: np.ndarray) -> None:
    """
    将 trajectory（pos+quat+vel+ang_vel）和 accel (steps,6) 保存到文件：
      data/ref_trajs/{name}_traj.npy
      data/ref_trajs/accel_{name}.npy

    traj.npy 存储为数组 shape (steps, 13):
      [pos_x,pos_y,pos_z, quat_w,quat_x,quat_y,quat_z, vel_x,vel_y,vel_z, angv_x,angv_y,angv_z]
    """
    steps = len(trajectory)
    arr = np.zeros((steps, 13), dtype=np.float32)
    for i, s in enumerate(trajectory):
        arr[i, 0:3]  = s["pos"]
        arr[i, 3:7]  = s["quat"]
        arr[i, 7:10] = s["vel"]
        arr[i, 10:13]= s["ang_vel"]
    traj_path = REF_DIR / f"{name}_traj.npy"
    np.save(traj_path, arr)
    print(f"Saved trajectory → {traj_path}  (shape={arr.shape})")

    accel_path = REF_DIR / f"accel_{name}.npy"
    np.save(accel_path, accel)
    print(f"Saved acceleration → {accel_path}  (shape={accel.shape})")


# ———— 主程序 ———— #
def main():
    # 1) 保存域随机化配置
    path_dom = DATA_DIR / "domain_rand.yaml"
    with open(path_dom, "w", encoding="utf-8") as f:
        yaml.dump(DOMAIN_RAND, f, sort_keys=False)
    print(f"Generated domain randomization config → {path_dom}")

    # 2) 定义一组目标任务
    targets = [
        {
            "type": "hover",
            "name": "hover",
            "pos": [0.0, 0.0, -1.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "type": "circle",
            "name": "circle",
            "radius": 1.0,
            "pos_z": -1.0,
        },
        {
            "type": "z_wave",
            "name": "z_wave",
            "amplitude": 0.5,
            "offset_z": -1.0,
            "pos_xy": [0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "type": "point_to_point",
            "name": "ptp_forward",
            "start": [0.0, 0.0, -1.0],
            "end": [3.0, 0.0, -1.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "type": "inspection",
            "name": "inspect_pipe",
            "pos": [1.5, 1.5, -3.0],
            "quat_sequence": [
                [1.0, 0.0, 0.0, 0.0],
                Rotation.from_euler("z", 45, degrees=True).as_quat()[[3, 0, 1, 2]],
                Rotation.from_euler("z", -45, degrees=True).as_quat()[[3, 0, 1, 2]],
            ],
        },
    ]

    # 3) 逐个生成并保存
    for tgt in targets:
        name = tgt["name"]
        traj = generate_feasible_trajectory(tgt)
        accel = compute_physical_acceleration(traj)
        save_trajectory(name, traj, accel)


if __name__ == "__main__":
    main()
