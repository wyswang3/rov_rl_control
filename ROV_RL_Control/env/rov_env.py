# 文件路径：rov_rl_control/env/rov_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from sim.parameters import ROVParameters
from sim.hydro_dynamics import ROVDynamics
from sim.thruster_allocation import ThrusterAllocator


class ROVEnv(gym.Env):
    """
    自定义 ROV 强化学习环境，目标在于：
      1. 快速将 ROV 移动到指定目标位置
      2. 在目标位置悬停时，能够抵御捕捞等外部扰动，保持稳定

    Observation (维度 25)：
      - 位置 (x, y, z)                  → 3
      - 姿态四元数 (qx, qy, qz, qw)       → 4
      - 线速度 (u, v, w)                → 3
      - 角速度 (p, q, r)                → 3
      - 深度 (z)                        → 1
      - 目标位置误差 (dx, dy, dz)       → 3
      - 上一次动作（功率比例，共 8）    → 8

    Action (维度 8, 范围 [0, 1])：
      - 每个值表示该推进器输出的功率比例，实际功率 P_i = action[i] * P_max
      - 推力 F_i = k * (P_i)^n（实验拟合关系：F = 1.5801 * P^0.5819）

    Reward 由以下几部分组成：
      - 位置误差惩罚：−k_pos * ||pos − target_pos||
      - 能耗惩罚：−k_energy * Σ P_i  （用功率而非直接用推力做惩罚）
      - 平滑惩罚：−k_smooth * ||action − prev_action||
      - 悬停额外奖励：当与目标足够接近且速度足够小时，给予 hover_bonus
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        disturbance_enable: bool = True,
    ) -> None:
        super().__init__()

        # -----------------------------
        # 1. 加载 ROV 物理参数 与 模块
        # -----------------------------
        self.params = ROVParameters()
        self.dynamics = ROVDynamics(self.params)
        self.thruster_allocator = ThrusterAllocator(self.params)

        # -----------------------------
        # 2. 定义动作空间 (Action Space)
        #    动作为 8 维功率比例，范围 [0,1]
        # -----------------------------
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # -----------------------------
        # 3. 定义观测空间 (Observation Space)
        #    观测维度：
        #      位置(3) + 四元数(4) + 线速度(3) + 角速度(3) + 深度(1)
        #      + 目标位置误差(3) + 上一次动作(8) = 25
        # -----------------------------
        obs_dim = 3 + 4 + 3 + 3 + 1 + 3 + 8
        obs_low = np.full((obs_dim,), -np.inf, dtype=np.float32)
        obs_high = np.full((obs_dim,), np.inf, dtype=np.float32)
        # 四元数约束在 [-1,1]
        obs_low[3:7] = -1.0
        obs_high[3:7] = 1.0
        # 动作历史约束在 [0,1]
        obs_low[-8:] = 0.0
        obs_high[-8:] = 1.0

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # -----------------------------
        # 4. 环境内部参数 与 初始状态
        # -----------------------------
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.disturbance_enable = disturbance_enable

        # 状态向量：pos(3) + quat(4) + lin_vel(3) + ang_vel(3) = 13
        self.state = np.zeros(13, dtype=np.float32)
        # 上一次动作（功率比例）
        self.prev_action = np.zeros(8, dtype=np.float32)
        # 目标位姿：{"position": np.ndarray(3,), "quat": np.ndarray(4,)}
        self.target_pose: Dict[str, np.ndarray] = {
            "position": np.zeros(3, dtype=np.float32),
            "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
        self.current_step = 0

        # -----------------------------
        # 5. 渲染相关（可选）
        # -----------------------------
        if self.render_mode == "human":
            # TODO: 初始化可视化资源（matplotlib、ROS 等）
            pass

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境：
          - 将 ROV 状态归零（位置原点，姿态单位四元数，速度 0）
          - 清零上一次动作
          - 随机或固定设定目标位姿
          - 返回初始观测
        """
        super().reset(seed=seed)
        self.current_step = 0

        # 1) 初始化 ROV 状态
        #    pos = [0,0,0], quat = [0,0,0,1], lin_vel = [0,0,0], ang_vel = [0,0,0]
        self.state[:] = 0.0
        self.state[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.prev_action[:] = 0.0

        # 2) 设定目标位姿（示例：固定在 (2,0,-1)）
        self.target_pose["position"] = np.array([2.0, 0.0, -1.0], dtype=np.float32)
        self.target_pose["quat"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # 3) 返回观察值
        observation = self._get_observation()
        return observation, {}

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        环境步进：
          1. 裁剪并解析动作（功率比例 ∈ [0,1]）
          2. 计算实际推力（F = k * (P_max * action)^n）
          3. 可选地注入外部扰动（捕捞拉力、随机洋流等）
          4. 调用动力学一步仿真（更新 state）
          5. 计算观测、奖励、是否结束
        返回：
          - observation: np.ndarray, 观测向量 (25,)
          - reward: float
          - done: bool
          - truncated: bool
          - info: dict
        """
        # 1) 裁剪动作到 [0,1]
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 2) 动作 → 推力
        #    ThrusterAllocator.map_action_to_thrust 会根据 F = k * (P)^n 计算推力
        thrusts = self.thruster_allocator.map_action_to_thrust(action)

        # 3) 注入外部扰动（如果开启）
        if self.disturbance_enable:
            disturbance_force, disturbance_torque = self._apply_disturbance()
        else:
            disturbance_force = np.zeros(3, dtype=np.float32)
            disturbance_torque = np.zeros(3, dtype=np.float32)

        # 4) 用动力学模块计算下一状态
        next_state = self.dynamics.step(
            state=self.state,
            thrusts=thrusts,
            disturbance_force=disturbance_force,
            disturbance_torque=disturbance_torque
        )

        # 5) 更新内部状态
        self.state[:] = next_state
        self.prev_action[:] = action
        self.current_step += 1

        # 6) 计算观测、奖励、终止条件
        obs = self._get_observation()
        reward = self._compute_reward(state=self.state, action=action)

        done = False
        truncated = False
        # 到达最大步数则 truncated
        if self.current_step >= self.max_episode_steps:
            truncated = True
        # 如果 ROV 超出安全边界（例如浮出水面 z > 0）则 done
        if self.state[2] > 0.0:
            done = True

        info: Dict[str, Any] = {
            "step": self.current_step,
            "position": self.state[0:3].copy(),
            "velocity": self.state[7:10].copy(),
        }

        return obs, reward, done, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        构造观测向量：
          [pos(3), quat(4), lin_vel(3), ang_vel(3), depth(1), pos_error(3), prev_action(8)]
        """
        pos = self.state[0:3]
        quat = self.state[3:7]
        lin_vel = self.state[7:10]
        ang_vel = self.state[10:13]

        depth = np.array([pos[2]], dtype=np.float32)  # z 轴，负向为水下
        pos_error = self.target_pose["position"] - pos  # 目标位置误差

        obs = np.concatenate([
            pos,
            quat,
            lin_vel,
            ang_vel,
            depth,
            pos_error,
            self.prev_action
        ], axis=0).astype(np.float32)

        return obs

    def _compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        计算奖励：
          - 位置误差惩罚：−k_pos * ||pos_error||
          - 能耗惩罚：−k_energy * Σ (P_i)， 其中 P_i = action[i] * P_max
          - 平滑惩罚：−k_smooth * ||action − prev_action||
          - 悬停奖励：当距离目标 < 0.1 且线速 < 0.05 且角速 < 0.05 时，+hover_bonus
        """
        pos = state[0:3]
        # 1) 位置误差
        pos_error = self.target_pose["position"] - pos
        dist_error = np.linalg.norm(pos_error)

        # 2) 悬停检查
        lin_vel = state[7:10]
        ang_vel = state[10:13]
        is_near = dist_error < 0.1
        is_slow = (np.linalg.norm(lin_vel) < 0.05) and (np.linalg.norm(ang_vel) < 0.05)
        hover_bonus = 1.0 if (is_near and is_slow) else 0.0

        # 3) 能耗惩罚：按功率计算
        #    P_i = action[i] * P_max
        P = action * self.params.P_max  # shape=(8,)
        energy_penalty = np.sum(P)

        # 4) 平滑惩罚
        smooth_penalty = np.linalg.norm(action - self.prev_action)

        # 5) 权重系数
        k_pos = 1.0
        k_energy = 0.001    # 能耗惩罚权重，可根据实验调整
        k_smooth = 0.1     # 平滑惩罚权重
        k_hover = 10.0     # 悬停奖励权重，可视需求增大

        reward = (-k_pos * dist_error
                  -k_energy * energy_penalty
                  -k_smooth * smooth_penalty
                  +k_hover * hover_bonus)

        return float(reward)

    def _apply_disturbance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        外部扰动模型：模拟捕捞时的拉力和随机洋流扰动。
        返回：
          - disturbance_force (3,), 单位 N
          - disturbance_torque (3,), 单位 N·m
        """
        # 1) 随机小洋流扰动
        current_force = np.array([
            np.random.normal(0.0, 0.3),
            np.random.normal(0.0, 0.3),
            0.0
        ], dtype=np.float32)

        # 2) 捕捞拉力：当距离目标 < 0.2 m 时施加
        pos = self.state[0:3]
        dist_to_target = np.linalg.norm(self.target_pose["position"] - pos)
        fishing_force = np.zeros(3, dtype=np.float32)
        fishing_torque = np.zeros(3, dtype=np.float32)
        if dist_to_target < 0.2:
            # 从 ROV 指向目标的方向，施加 5 N 拉力
            direction = (self.target_pose["position"] - pos)
            direction_norm = np.linalg.norm(direction) + 1e-8
            unit_dir = direction / direction_norm
            fishing_force = -5.0 * unit_dir
            # 随机小扭矩
            fishing_torque = np.random.normal(0.0, 0.05, size=(3,)).astype(np.float32)

        disturbance_force = current_force + fishing_force
        disturbance_torque = fishing_torque
        return disturbance_force, disturbance_torque

    def render(self) -> None:
        """
        可视化接口（若需要），例如使用 matplotlib 绘制当前位姿或路径。
        """
        if self.render_mode == "human":
            # TODO: 根据需求实现可视化
            pass

    def close(self) -> None:
        """
        清理资源，如关闭图形窗口
        """
        if self.render_mode == "human":
            # TODO: 关闭渲染资源
            pass


# 注册环境，方便使用 gymnasium.make("ROV-v0")
gym.register(
    id="ROV-v0",
    entry_point="env.rov_env:ROVEnv",
    max_episode_steps=1000
)
