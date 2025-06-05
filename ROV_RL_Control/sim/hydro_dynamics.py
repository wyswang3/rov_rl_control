# 文件路径：rov_rl_control/sim/hydro_dynamics.py

import numpy as np
from typing import Any, Dict, Tuple

from sim.parameters import ROVParameters

class ROVDynamics:
    """
    ROVDynamics 类负责水下机器人在给定当前状态、推进器推力（及扰动）时，
    计算下一时刻的运动状态。基于 6-DOF 理论水动力学方程：

        M * ν̇ + C(ν) * ν + D(ν) * ν + g(η) = τ_total

    其中：
      - ν = [u, v, w, p, q, r]^T，线速度与角速度
      - η = [x, y, z, φ, θ, ψ]^T，位置与姿态（可通过四元数转换）
      - M：总质量矩阵（含刚体与附加质量）
      - C(ν)：科氏力矩阵
      - D(ν)：阻尼矩阵（与速度相关的线性/非线性阻尼）
      - g(η)：重力与浮力项（根据姿态计算合力）
      - τ_total：外部驱动力矩，包括推进器推力与水流/捕捞扰动

    设计目标：
      1. 通过简单明了的离散化公式，将连续动力学转换到每个时间步 dt。
      2. 物理计算尽量利用 NumPy 向量/矩阵运算，避免 Python 循环。
      3. 保持结构清晰，将 M、C、D、g 的计算拆分为独立方法，便于后续调参或扩展。
    """

    def __init__(self, params: ROVParameters) -> None:
        """
        初始化动力学模块：
          - params：ROVParameters 实例，包含质量、惯量、阻尼、附加质量、浮力、重心等
          - 从 params 中提取所需矩阵和标量：M_rb、M_a、D、重力、浮力等
        """
        self.params = params

        # 刚体质量矩阵 M_rb (6×6) 包括质量和惯性矩阵
        self.M_rb = self.params.M_rb  # 6×6 numpy.ndarray
        # 附加质量矩阵 M_a (6×6)
        self.M_a = self.params.M_a    # 6×6 numpy.ndarray
        # 总质量矩阵 M = M_rb + M_a
        self.M = self.M_rb + self.M_a # 6×6 numpy.ndarray
        # 阻尼矩阵 D (6×6)，可以是线性+非线性组合
        self.D = self.params.D        # 6×6 numpy.ndarray
        # 重力与浮力项的参数
        self.weight = self.params.weight      # 标量，重力 (N)
        self.buoyancy = self.params.buoyancy  # 标量，浮力 (N)
        self.center_of_gravity = self.params.center_of_gravity  # 3-vector (m)
        self.center_of_buoyancy = self.params.center_of_buoyancy  # 3-vector (m)

        # 时间步长 dt
        self.dt = self.params.dt  # 单位：秒

        # 预分配临时变量，减少重复分配
        self._nu = np.zeros(6, dtype=np.float32)       # [u, v, w, p, q, r]
        self._eta = np.zeros(6, dtype=np.float32)      # [x, y, z, φ, θ, ψ] （欧拉角）
        self._p = np.zeros((6,), dtype=np.float32)     # 总力/力矩
        self._dot_nu = np.zeros((6,), dtype=np.float32)  # 加速度

    def step(
            self,
            state: np.ndarray,
            thrusts: np.ndarray,
            disturbance_force: np.ndarray,
            disturbance_torque: np.ndarray
    ) -> np.ndarray:
        """
        核心步进函数：
          - state: 当前状态向量 [x, y, z, qx, qy, qz, qw, u, v, w, p, q, r]
            其中：位置 + 四元数 + 线速度 + 角速度，长度为 3 + 4 + 3 + 3 = 13
          - thrusts: 8 推进器的推力 numpy.ndarray (8,)
          - disturbance_force: 外部扰动力 3-vector (单位 N)
          - disturbance_torque: 外部扰矩阵 3-vector (单位 N·m)

        返回：
          - next_state: 下一个时刻的状态向量，结构与输入 state 相同
        """
        # 1) 先将四元数转换为欧拉角，便于计算重力/浮力 g(η)
        #    然后将所有状态分解到 ν (6)和 η (6) 中
        pos = state[0:3]           # x, y, z
        quat = state[3:7]          # qx, qy, qz, qw
        nu_lin = state[7:10]       # u, v, w
        nu_ang = state[10:13]      # p, q, r

        # 四元数 → 欧拉角 (roll, pitch, yaw)
        # 以下函数可用 scipy 或手动实现；此处简化为伪代码
        phi, theta, psi = self._quat_to_euler(quat)
        self._nu[:] = np.concatenate([nu_lin, nu_ang])
        self._eta[:] = np.concatenate([pos, np.array([phi, theta, psi], dtype=np.float32)])

        # 2) 计算总驱动力/力矩 p = τ_thrust + τ_disturb - τ_gravity_buoyancy - C(ν)ν - D(ν)ν
        #   2.1) 推进器驱动力矩 (6×1)：先把 8 个推力映射到 6-DOF → thrust_vector_6
        thrust_vector_6 = self._map_thrusts_to_force_moment(thrusts)

        #   2.2) 外部扰动力矩 (6×1)：前 3 为力，后 3 为矩
        disturbance_6 = np.concatenate([disturbance_force, disturbance_torque], axis=0)

        #   2.3) 重力与浮力对力矩 g_6 (6×1)：只与姿态相关
        g_6 = self._compute_gravity_buoyancy(phi, theta, psi)

        #   2.4) 科氏力矩矩阵 C(ν) (6×6) 乘以 ν (6×1)
        C_nu = self._compute_coriolis_matrix(self._nu) @ self._nu

        #   2.5) 阻尼 D(ν) (6×6) 乘以 ν (6×1)
        D_nu = self.D @ self._nu  # 假设阻尼矩阵 D 已经包含线性/非线性项

        # 总力/力矩向量 p = thrust_vector_6 + disturbance_6 - g_6 - C_nu - D_nu
        self._p[:] = (thrust_vector_6
                      + disturbance_6
                      - g_6
                      - C_nu
                      - D_nu)

        # 3) 计算 ν̇ = M^{-1} * p
        #    M (6×6) 在 __init__ 已预先求好
        self._dot_nu[:] = np.linalg.solve(self.M, self._p)

        # 4) 更新 ν ← ν + ν̇ * dt
        nu_next = self._nu + self._dot_nu * self.dt

        # 5) 更新 η（位置、姿态）：
        #    位置更新：pos_next = pos + R(η) @ ν_lin_next * dt
        #    姿态更新：quat_next = quat ⊗ quat_dot * dt；quat_dot = 0.5 * Ω(ν_ang_next) ⊗ quat
        #    这里用小角度近似或四元数积分
        pos_next = pos + self._rotation_matrix(phi, theta, psi) @ nu_next[0:3] * self.dt
        quat_next = self._quaternion_integration(quat, nu_next[3:6], self.dt)

        # 6) 拼接 next_state：pos_next(3) + quat_next(4) + nu_lin_next(3) + nu_ang_next(3)
        next_state = np.zeros_like(state)
        info: Dict[str, Any] = {
            "debug": True
        }
        return next_state

    def _quat_to_euler(self, quat: np.ndarray) -> Tuple[float, float, float]:
        """
        将四元数 [qx, qy, qz, qw] 转换为欧拉角 (φ, θ, ψ)。
        示例使用常见转换公式，注意顺序：ZYX 或 XYZ，视你的坐标系定义而定。
        """
        qx, qy, qz, qw = quat
        # 下面示例为 ZYX 顺序（航向-俯仰-横滚），可根据实际需要调整
        # roll (φ)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        phi = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (θ)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            theta = np.sign(sinp) * (np.pi / 2)  # 90 度
        else:
            theta = np.arcsin(sinp)

        # yaw (ψ)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        psi = np.arctan2(siny_cosp, cosy_cosp)

        return float(phi), float(theta), float(psi)

    def _rotation_matrix(
        self,
        phi: float,
        theta: float,
        psi: float
    ) -> np.ndarray:
        """
        根据欧拉角 (φ, θ, ψ) 计算旋转矩阵 R(η)，将机体系速度映射到惯性系：
            R = R_z(ψ) * R_y(θ) * R_x(φ)
        返回 3×3 旋转矩阵
        """
        # Precompute cos/sin
        cφ = np.cos(phi); sφ = np.sin(phi)
        cθ = np.cos(theta); sθ = np.sin(theta)
        cψ = np.cos(psi); sψ = np.sin(psi)

        # 依次旋转：
        R_x = np.array([[1,    0,     0],
                        [0,   cφ,   -sφ],
                        [0,   sφ,    cφ]], dtype=np.float32)

        R_y = np.array([[cθ,   0,    sθ],
                        [ 0,   1,     0],
                        [-sθ,  0,    cθ]], dtype=np.float32)

        R_z = np.array([[cψ,  -sψ,   0],
                        [sψ,   cψ,   0],
                        [ 0,    0,   1]], dtype=np.float32)

        R = R_z @ R_y @ R_x
        return R

    def _compute_gravity_buoyancy(
        self,
        phi: float,
        theta: float,
        psi: float
    ) -> np.ndarray:
        """
        计算重力与浮力在机体系(6×1)上的投影：
          - 重力：向惯性系负 z 方向 (0, 0, -weight)
          - 浮力：向惯性系正 z 方向 (0, 0, buoyancy)
          根据姿态将力投影到机体系，并计算力矩

        返回 6×1 向量： [force_body(3), moment_body(3)]
        """
        # 1) 惯性系中的重力与浮力
        fg_inertial = np.array([0.0, 0.0, -self.weight], dtype=np.float32)
        fb_inertial = np.array([0.0, 0.0, self.buoyancy], dtype=np.float32)

        # 2) 将其从惯性系转换到机体系： F_body = R(φ,θ,ψ).T @ F_inertial
        R = self._rotation_matrix(phi, theta, psi)
        fg_body = R.T @ fg_inertial
        fb_body = R.T @ fb_inertial

        # 3) 计算不平衡力矩： M = (r_buoyancy × Fb_body) - (r_gravity × Fg_body)
        #    这里 r_buoyancy = center_of_buoyancy， r_gravity = center_of_gravity
        moment_buoy = np.cross(self.center_of_buoyancy, fb_body)
        moment_grav = np.cross(self.center_of_gravity, fg_body)
        # 总力 = Fb_body + Fg_body；总力矩 = moment_buoy + moment_grav
        force_body = fb_body + fg_body
        moment_body = moment_buoy + moment_grav

        return np.concatenate([force_body, moment_body], axis=0).astype(np.float32)

    def _compute_coriolis_matrix(self, nu: np.ndarray) -> np.ndarray:
        """
        计算科氏力矩阵 C(ν)，这里只举例基于附加质量 M_a 的线性近似：
          C_a(ν) = [[ 0, -m_a*w,  m_a*v, ... ], ... ]  (6×6)
        然后 C(ν) = C_rb(ν) + C_a(ν)，其中 C_rb(ν) 基于刚体惯性计算。

        具体公式较为冗长，此处可先实现简单近似：
          C(ν) ≈ skew(momentum)  例如 C(ν) @ ν ≈ cross( M * ν, ν )
        如果暂不需要完整 C 矩阵，可先置零矩阵或只计算阻尼。
        """
        # 简化：先用零矩阵替代
        return np.zeros((6, 6), dtype=np.float32)

    def _map_thrusts_to_force_moment(self, thrusts: np.ndarray) -> np.ndarray:
        """
        将 8 个推进器的推力值映射到 6-DOF 的力和力矩向量 τ_thrust：
          τ_thrust = B * thrusts   (B 为 6×8 的推力分配矩阵)
        其中 B 由参数提供（在 ROVParameters 中定义）。
        """
        # B 矩阵 (6×8)
        B = self.params.thrust_allocation_matrix  # numpy.ndarray (6×8)
        return B @ thrusts

    def _quaternion_integration(
        self,
        quat: np.ndarray,
        omega: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        用四元数微分方程更新姿态：
          quat_dot = 0.5 * Ω(ω) @ quat
          quat_next = quat + quat_dot * dt
          然后归一化 quat_next

        其中 Ω(ω) 为 4×4 矩阵：
            [ 0,   -p,   -q,   -r ]
            [ p,    0,   r,   -q ]
            [ q,   -r,    0,    p ]
            [ r,    q,   -p,    0 ]

        输入：
          - quat: 当前四元数 (4,)
          - omega: 当前角速度 [p, q, r] (3,)
          - dt: 时间步长

        返回：
          - quat_next: 归一化后下一时刻四元数 (4,)
        """
        p, q, r = omega
        qx, qy, qz, qw = quat

        # 构建 Ω(ω)
        Omega = np.array([
            [ 0.0, -p,   -q,   -r ],
            [ p,    0.0,  r,   -q ],
            [ q,   -r,    0.0,  p ],
            [ r,    q,   -p,    0.0 ]
        ], dtype=np.float32)

        quat_dot = 0.5 * Omega @ quat.reshape(4, 1)
        quat_dot = quat_dot.reshape(4)

        quat_next = quat + quat_dot * dt
        # 归一化
        norm = np.linalg.norm(quat_next) + 1e-8
        quat_next = quat_next / norm

        return quat_next.astype(np.float32)
