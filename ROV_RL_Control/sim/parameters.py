# sim/parameters.py
import numpy as np
from typing import Any, Dict

class ROVParameters:
    """
    ROVParameters 类封装了水下机器人仿真中所需的所有物理参数，
    以及坐标和姿态相关的阈值、初始状态。
    """
    def __init__(self) -> None:
        # -----------------------------
        # 1. 基础物理量
        # -----------------------------
        self.m = 13.5                    # 质量 [kg]
        self.g = 9.82                   # 重力加速度 [m/s^2]
        self.rho = 1000.0               # 水密度 [kg/m^3]
        self.vol = 0.0134               # 排水体积 [m^3]
        self.buoyancy = self.rho * self.g * self.vol  # 浮力 [N]

        # -----------------------------
        # 2. 惯性矩阵和附加质量
        # -----------------------------
        # 刚体惯性矩阵 I_body (3×3)
        I_body_vals = np.array([0.26, 0.23, 0.37], dtype=np.float32)
        I_body = np.diag(I_body_vals)
        # 附加质量 M_a (6×6)
        Ma_vals = np.array([6.36, 7.12, 18.68, 0.189, 0.135, 0.222], dtype=np.float32)
        self.M_a = np.diag(Ma_vals)
        # 刚体质量矩阵 M_rb
        I3 = np.eye(3, dtype=np.float32)
        top = np.hstack((self.m * I3, np.zeros((3,3), dtype=np.float32)))
        bottom = np.hstack((np.zeros((3,3), dtype=np.float32), I_body))
        self.M_rb = np.vstack((top, bottom))
        # 总质量矩阵 M 和逆矩阵
        self.M = self.M_rb + self.M_a
        self.M_inv = np.linalg.inv(self.M)

        # -----------------------------
        # 3. 阻尼矩阵 D
        # -----------------------------
        # 线性阻尼
        D_vals = np.array([13.7, 22.0, 33.0, 0.4, 0.8, 0.5], dtype=np.float32)
        self.D = -np.diag(D_vals)
        # 非线性阻尼
        Dn_vals = np.array([141.0, 217.0, 190.0, 1.19, 0.47, 1.5], dtype=np.float32)
        self.Dn = -np.diag(Dn_vals)

        # -----------------------------
        # 4. 重心与浮心坐标
        # -----------------------------
        self.r_g = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.r_b = np.array([0.0, 0.0, 0.01], dtype=np.float32)
        self.center_of_gravity = self.r_g
        self.center_of_buoyancy = self.r_b

        # -----------------------------
        # 5. 推力分配矩阵及伪逆
        # -----------------------------
        self.thrust_allocation_matrix = np.array([
            [ 0.7071,  0.7071, -0.7071, -0.7071,  0.0,    0.0,    0.0,   0.0],
            [-0.7071,  0.7071, -0.7071,  0.7071,  0.0,    0.0,    0.0,   0.0],
            [ 0.0,     0.0,     0.0,     0.0,    -1.0,    1.0,    1.0,  -1.0],
            [ 0.0,     0.0,     0.0,     0.0,    0.218,   0.218,  -0.218,-0.218],
            [ 0.0,     0.0,     0.0,     0.0,     0.12,   -0.12,    0.12,-0.12],
            [-0.1888,  0.1888,  0.1888, -0.1888,  0.0,     0.0,     0.0,   0.0]
        ], dtype=np.float32)
        self.thrust_allocation_pinv = np.linalg.pinv(self.thrust_allocation_matrix)
        self.max_thrusts = np.full((8,), 50.0, dtype=np.float32)

        # -----------------------------
        # 6. 初始状态 & 阈值
        # -----------------------------
        self.init_position = np.zeros(3, dtype=np.float32)
        self.init_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.init_velocity = np.zeros(3, dtype=np.float32)
        self.init_omega = np.zeros(3, dtype=np.float32)
        self.angle_threshold = np.pi

        # -----------------------------
        # 7. 时间步长 dt
        # -----------------------------
        self.dt = 1.0 / 50.0

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k,v in self.__dict__.items()}
