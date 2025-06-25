# sim/thruster_allocation.py
import numpy as np
from typing import Optional

from sim.parameters import ROVParameters

class ThrusterAllocator:
    """
    ThrusterAllocator 将 RL 输出或目标力/力矩映射到推进器推力，支持：
      1. 将 0-1 归一化动作映射为实际推力
      2. 将 6-DOF 目标力/力矩最小二乘分配到 8 个推进器
    """
    def __init__(
        self,
        params: Optional[ROVParameters] = None
    ) -> None:
        self.params = params or ROVParameters()
        # 最大功率 (W)
        self.P_max = 240.0
        # 功率-推力拟合参数: F = k * P^n
        self.k = 1.5801
        self.n = 0.5819
        # 推力限制
        self.max_thrusts = self.params.max_thrusts  # shape (8,)
        # 分配矩阵
        self.T = self.params.thrust_allocation_matrix  # shape (6,8)
        self.T_pinv = self.params.thrust_allocation_pinv  # shape (8,6)

    def map_action_to_thrust(self, action: np.ndarray) -> np.ndarray:
        """
        将 action ∈ [0,1]^8 映射为推力 (N)
        """
        a = np.clip(action, 0.0, 1.0).astype(np.float32)
        # 功率
        P = a * self.P_max
        # 推力
        thrusts = self.k * np.power(P, self.n)
        # 限幅
        thrusts = np.clip(thrusts, 0.0, self.max_thrusts)
        return thrusts

    def map_force_to_action(self, force_moment: np.ndarray, clamp: bool = True) -> np.ndarray:
        """
        将 6DOF 力/力矩映射回 8 个归一化动作 ∈ [0,1]
        """
        # 分配到推力
        thrusts = self.T_pinv.dot(force_moment.astype(np.float32))
        if clamp:
            thrusts = np.clip(thrusts, 0.0, self.max_thrusts)
        # 反推功率
        thrusts = np.clip(thrusts, 0.0, None)
        P = np.power(thrusts / self.k, 1.0 / self.n)
        P = np.clip(P, 0.0, self.P_max)
        # 归一化
        action = P / self.P_max
        return action.astype(np.float32)

    def allocate(self, action: np.ndarray) -> np.ndarray:
        """
        直接从 RL action 到 6DOF 力矩: action->thrusts->force_moment
        """
        thrusts = self.map_action_to_thrust(action)
        force_moment = self.T.dot(thrusts)
        return force_moment

    def inverse_allocation_error(self, force_moment: np.ndarray) -> float:
        """
        评估 T @ (T_pinv @ force_moment) 与 force_moment 的误差
        """
        recon = self.T.dot(self.T_pinv.dot(force_moment))
        return float(np.linalg.norm(recon - force_moment))
