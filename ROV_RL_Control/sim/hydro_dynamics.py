# sim/hydro_dynamics.py
"""
ROV 六自由度刚体水动力学模型 (基于四元数)
动力学方程: M * dot(v) + C(v)*v + D(v)*v + g(eta) = tau
state = [pos(3), quat(4), v(3), omega(3)]
其中 quat = [qw, qx, qy, qz]
"""
import numpy as np
from sim.parameters import ROVParameters

class HydroDynamics:
    def __init__(self, params: ROVParameters = None):
        # 初始化物理参数
        self.params = params or ROVParameters()
        self.M_inv = self.params.M_inv         # 6×6 质量矩阵逆
        self.C = None  # 可自定义Coriolis项
        self.D = None  # 可自定义阻尼项
        self.dt = self.params.dt

    def quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
        # 将四元数转为旋转矩阵
        qw, qx, qy, qz = q
        # 归一化
        norm = np.linalg.norm(q)
        qw, qx, qy, qz = q / norm
        # 旋转矩阵计算
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ], dtype=np.float32)
        return R

    def omega_to_quat_mat(self, omega: np.ndarray) -> np.ndarray:
        # 构造用于四元数导数的矩阵: Omega(omega)
        wx, wy, wz = omega
        return np.array([
            [ 0.0, -wx,  -wy,  -wz ],
            [ wx,   0.0,  wz,  -wy ],
            [ wy,  -wz,   0.0,  wx ],
            [ wz,   wy,  -wx,  0.0 ]
        ], dtype=np.float32)

    def acceleration(self, v_full: np.ndarray, tau: np.ndarray) -> np.ndarray:
        # 计算6维速度加速度: v_full = [u,v,w,p,q,r]
        # 可扩展 C(v) 和 D(v) 模型
        # 简化：仅线性阻尼 + 重力浮力
        g = np.zeros(6, dtype=np.float32)
        Fz = self.params.m * self.params.g - self.params.buoyancy
        g[2] = Fz
        rhs = tau - g
        return self.M_inv.dot(rhs)

    def deriv(self, state: np.ndarray, tau: np.ndarray) -> np.ndarray:
        # 计算状态导数
        pos = state[0:3]             # 位置
        quat = state[3:7]            # 四元数
        v_lin = state[7:10]          # 线速度
        omega = state[10:13]         # 角速度
        # 位置导数: dot(pos) = R(q) * v_lin
        R = self.quaternion_to_rotation(quat)
        dot_pos = R.dot(v_lin)
        # 四元数导数: dot(quat) = 0.5 * Omega(omega) * quat
        Omega = self.omega_to_quat_mat(omega)
        dot_quat = 0.5 * Omega.dot(quat)
        # 6维加速度
        v_full = np.concatenate([v_lin, omega])
        dot_vfull = self.acceleration(v_full, tau)
        # 拆分线性与角加速度
        dot_v_lin = dot_vfull[0:3]
        dot_omega = dot_vfull[3:6]
        # 拼接返回 3 + 4 + 3 + 3 = 13 维导数
        return np.concatenate([dot_pos, dot_quat, dot_v_lin, dot_omega])

    def integrate(self, state: np.ndarray, tau: np.ndarray) -> np.ndarray:
        # 使用 RK4 进行数值积分
        h = self.dt
        k1 = self.deriv(state, tau)
        k2 = self.deriv(state + 0.5*h*k1, tau)
        k3 = self.deriv(state + 0.5*h*k2, tau)
        k4 = self.deriv(state + h*k3, tau)
        next_state = state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # 确保四元数归一化
        next_state[3:7] /= np.linalg.norm(next_state[3:7])
        return next_state
