# 文件：rov_rl_control/sim/thruster_allocation.py

import numpy as np
from typing import Optional

from sim.parameters import ROVParameters


class ThrusterAllocator:
    """
    ThrusterAllocator 负责：
      1. 将 RL Agent 输出的归一化动作 (action ∈ [0, 1]^8) 映射到实际推力值 (thrusts ∈ R^8)
         - 首先 P_i = action[i] * P_max
         - 然后 F_i = k * P_i^n （实验拟合关系：F = 1.5801 * P^0.5819）
      2. （可选）将 6-DOF 目标合力/力矩 (force_moment ∈ R^6) 通过伪逆分配到 8 台推进器推力，
         并再反推对应功率/动作。适用于“先计算力/力矩再分配”的架构。
    """

    def __init__(self, params: ROVParameters) -> None:
        # 8 维最大功率，每台推进器最大不超过 240 W
        self.P_max = 240.0  # 单位：W
        # 实验拟合参数：F = k * P^n
        self.k = 1.5801
        self.n = 0.5819

        # 8 维最大推力（可根据推进器规格结合功率-推力曲线预估得到，也可不直接使用）
        # 这里先保留 max_thrusts（不一定要用到），以便与 6-DOF 正向矩阵配合使用
        self.max_thrusts = params.max_thrusts      # 8 维 numpy.ndarray

        # 6×8 推力分配矩阵 T（正向：thrusts_8 -> force_moment_6）
        self.T = params.thrust_allocation_matrix   # numpy.ndarray, shape=(6,8)
        # 伪逆矩阵 (8×6)
        self.T_pinv = params.thrust_allocation_pinv


    def map_action_to_thrust(self, action: np.ndarray) -> np.ndarray:
        """
        将 RL Agent 输出的归一化动作 action ∈ [0, 1]^8 转换为 8 个推进器的实际推力值 (单位 N)：
          1) P_i = action[i] * P_max
          2) thrust_i = k * (P_i) ^ n

        输入：
          - action: numpy.ndarray, shape=(8,), 元素范围 [0, 1]

        返回：
          - thrusts: numpy.ndarray, shape=(8,), 单位 N（>= 0）
        """
        # 1. 首先裁剪 action 到 [0, 1]
        clipped_act = np.clip(action, 0.0, 1.0)

        # 2. 计算每台推进器对应的实际功率 P_i ∈ [0, P_max]
        P = clipped_act * self.P_max  # shape: (8,)

        # 3. 根据实验拟合关系式计算推力 F_i
        #    F = k * P^n
        #    注意：P^n，当 P=0 时 F=0；P>0 时返回正值
        thrusts = self.k * np.power(P, self.n)

        # 4. 如果需要保证 thrusts 不超过机械极限，可做裁剪
        thrusts = np.clip(thrusts, 0.0, self.max_thrusts)

        return thrusts.astype(np.float32)


    def map_force_to_thrusters(
        self,
        force_moment: np.ndarray,
        clamp: bool = True
    ) -> np.ndarray:
        """
        将给定的 6-DOF 目标合力/合力矩 force_moment ∈ R^6 映射到 8 台推进器的推力 (thrusts ∈ R^8)：
          thrusts = T_pinv @ force_moment  （最小二乘解）
          如果 clamp=True，则将 thrusts 裁剪到 [0, max_thrusts]，
          然后再反推对应功率 action = (P_thrust / P_max)^(1/n)。

        注意：此方法假设推力方向均为正向（本例中忽略反向推力），若推进器支持正反向，则需将动作范围设为 [-1, 1] 并对功率取绝对值。

        参数：
          - force_moment: numpy.ndarray, shape=(6,), 6-DOF 目标力和力矩
          - clamp: bool，是否在伪逆解后对 thrusts 进行裁剪

        返回：
          - action: numpy.ndarray, shape=(8,), 推力映射回的归一化功率动作 ∈ [0, 1]
        """
        # 1. 最小二乘解：8 维推力
        thrusts = self.T_pinv @ force_moment  # shape: (8,)

        if clamp:
            thrusts = np.clip(thrusts, 0.0, self.max_thrusts)

        # 2. 将推力反向映射到功率：P_i = (F_i / k)^(1/n)
        #    若 F_i=0，则 P_i=0；注意避免分母为零或浮点数域问题
        #    先保证 thrusts >= 0
        thrusts = np.clip(thrusts, 0.0, None)
        # 计算 P，再裁剪到 [0, P_max]
        P = np.power(thrusts / self.k, 1.0 / self.n)
        P = np.clip(P, 0.0, self.P_max)

        # 3. 归一化功率动作 action = P / P_max ∈ [0, 1]
        action = P / self.P_max

        return action.astype(np.float32)


    def inverse_allocation_error(
        self,
        force_moment: np.ndarray
    ) -> float:
        """
        评估伪逆映射的重构误差：
          reconstructed = T @ (T_pinv @ force_moment)
          error = || reconstructed - force_moment ||_2

        返回：
          - error: float，二范数误差
        """
        thrusts = self.T_pinv @ force_moment
        reconstructed = self.T @ thrusts
        error = np.linalg.norm(reconstructed - force_moment)
        return float(error)
