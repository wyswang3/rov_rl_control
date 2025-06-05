# 文件路径：rov_rl_control/sim/parameters.py

import numpy as np
from typing import Any, Dict, Tuple

class ROVParameters:
    """
    ROVParameters 类封装了水下机器人仿真中所需的所有物理参数，
    这些参数将被 ROVDynamics 和 ThrusterAllocator 等模块引用。
    """

    def __init__(self) -> None:
        # -----------------------------
        # 1. 基础物理量
        # -----------------------------
        # 质量 [kg]
        self.m = 13.5
        # 重力加速度 [m/s^2]
        self.g = 9.82
        # 水密度 [kg/m^3]
        self.rho = 1000.0
        # 排水体积 [m^3]
        self.vol = 0.0134
        # 浮力 B = ρ g V [N]
        self.buoyancy = self.rho * self.g * self.vol

        # -----------------------------
        # 2. 惯性矩阵和附加质量
        # -----------------------------
        # 刚体惯性矩阵 I (仅角动量部分) [kg·m^2]
        #   在 body frame 下，表示绕 x, y, z 轴的转动惯量
        I_body = np.diag([0.26, 0.23, 0.37], dtype=np.float32)  # 3×3

        # 附加质量 Mo (6×6)：对六自由度速度的附加惯性
        #   diag([Ma_u, Ma_v, Ma_w, Ma_p, Ma_q, Ma_r])
        self.M_a = np.diag([6.36, 7.12, 18.68, 0.189, 0.135, 0.222], dtype=np.float32)

        # 刚体质量矩阵 M_rb (6×6)：
        #   [ m * I3,       0  ]
        #   [    0,     I_body ]
        I3 = np.eye(3, dtype=np.float32)
        top = np.hstack((self.m * I3, np.zeros((3, 3), dtype=np.float32)))
        bottom = np.hstack((np.zeros((3, 3), dtype=np.float32), I_body))
        self.M_rb = np.vstack((top, bottom))  # 6×6

        # 总质量矩阵 M = M_rb + M_a (6×6)
        self.M = self.M_rb + self.M_a
        # 逆矩阵 M^{-1}
        self.M_inv = np.linalg.inv(self.M)

        # -----------------------------
        # 3. 阻尼矩阵 D
        # -----------------------------
        # 线性阻尼矩阵 D: diag([Xu, Yv, Zw, Kp, Mq, Nr])
        #   根据给定参数取负号（在动力学方程里为 D(ν)ν 项的一部分）
        self.D = -np.diag([13.7, 22.0, 33.0, 0.4, 0.8, 0.5], dtype=np.float32)

        # 非线性阻尼（如二次型）也可单独定义，这里用另一个矩阵保存
        # Dn: diag([Xuu, Yvv, Zww, Kpp, Mqq, Nrr])
        self.Dn = -np.diag([141.0, 217.0, 190.0, 1.19, 0.47, 1.5], dtype=np.float32)

        # -----------------------------
        # 4. 重力与浮力作用点（重心与浮心）
        # -----------------------------
        # 重心在机体坐标系下的坐标 [m]
        self.r_g = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # 浮心在机体坐标系下的坐标 [m]
        self.r_b = np.array([0.0, 0.0, 0.01], dtype=np.float32)
        # 浮心相对重心位置 r_cb = r_b - r_g
        self.center_of_buoyancy = self.r_b
        self.center_of_gravity = self.r_g

        # -----------------------------
        # 5. 推力分配矩阵 T (6×8) 及其伪逆 T_inv
        # -----------------------------
        # 8 台推进器在机体坐标系下的布置，对应 6-DOF 力/力矩的分配
        self.thrust_allocation_matrix = np.array([
            [ 0.7071,  0.7071, -0.7071, -0.7071,    0.0,    0.0,    0.0,    0.0],
            [-0.7071,  0.7071, -0.7071,  0.7071,    0.0,    0.0,    0.0,    0.0],
            [    0.0,       0.0,      0.0,      0.0,   -1.0,    1.0,    1.0,   -1.0],
            [    0.0,       0.0,      0.0,      0.0,  0.218,  0.218, -0.218, -0.218],
            [    0.0,       0.0,      0.0,      0.0,   0.12,  -0.12,   0.12,  -0.12],
            [-0.1888,  0.1888,  0.1888, -0.1888,    0.0,    0.0,    0.0,    0.0]
        ], dtype=np.float32)  # shape: (6, 8)

        # 伪逆矩阵，用于反向分配（如果需要解 6→8 的控制量）
        self.thrust_allocation_pinv = np.linalg.pinv(self.thrust_allocation_matrix)

        # -----------------------------
        # 6. 每台推进器的最大推力限制（N）
        # -----------------------------
        # 假设所有推进器最大推力相同，也可以逐个自定义
        # 这里示例值需要根据实际推进器规格确定。例如：
        max_thrust_per_thruster = 50.0  # 每台推进器最大推力约 50 N，可根据实际调整
        self.max_thrusts = np.full((8,), max_thrust_per_thruster, dtype=np.float32)

        # -----------------------------
        # 7. 时间步长 dt
        # -----------------------------
        # 假设环境的采样频率为 50 Hz，则 dt = 1 / 50
        self.dt = 1.0 / 50.0  # [s]

    def as_dict(self) -> Dict[str, Any]:
        """
        如果需要将所有参数打包为字典返回，可用此函数。
        """
        return {
            "m": self.m,
            "g": self.g,
            "rho": self.rho,
            "vol": self.vol,
            "buoyancy": self.buoyancy,
            "M_rb": self.M_rb,
            "M_a": self.M_a,
            "M": self.M,
            "M_inv": self.M_inv,
            "D": self.D,
            "Dn": self.Dn,
            "r_g": self.r_g,
            "r_b": self.r_b,
            "center_of_gravity": self.center_of_gravity,
            "center_of_buoyancy": self.center_of_buoyancy,
            "thrust_allocation_matrix": self.thrust_allocation_matrix,
            "thrust_allocation_pinv": self.thrust_allocation_pinv,
            "max_thrusts": self.max_thrusts,
            "dt": self.dt,
        }
