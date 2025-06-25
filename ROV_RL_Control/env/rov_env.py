# env/rov_env.py
"""
ROV 仿真环境，提供 Gym 接口（兼容 gym 和 gymnasium）

State: [pos(3), quat(4), lin_vel(3), ang_vel(3)]  共13维
Action: 各推进器推力 (8维)
"""
import numpy as np
from typing import Tuple, Optional

import gym
from gym import spaces, utils

from sim.parameters import ROVParameters
from sim.hydro_dynamics import HydroDynamics
from sim.thruster_allocation import ThrusterAllocator

class ROVEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        task: str = 'pose_control',
        params: Optional[ROVParameters] = None
    ):
        super().__init__()
        self.params = params or ROVParameters()
        self.dynamics = HydroDynamics(self.params)
        self.allocator = ThrusterAllocator(self.params)

        # 动作空间: 8台推进器推力
        max_thr = self.params.max_thrusts
        self.action_space = spaces.Box(
            low=-max_thr,
            high= max_thr,
            shape=(8,),
            dtype=np.float32
        )

        # 观测空间: pos(3), quat(4), lin_vel(3), ang_vel(3)
        inf = np.finfo(np.float32).max
        obs_high = np.array([
            *[inf]*3,
            *[1.0]*4,
            *[inf]*3,
            *[inf]*3
        ], dtype=np.float32)
        self.observation_space = spaces.Box(
            -obs_high,
             obs_high,
             dtype=np.float32
        )

        assert task in ('pose_control', 'path_following'), f"Unsupported task: {task}"
        self.task = task
        self.seed()

    def seed(self, seed: Optional[int] = None) -> int:
        self.np_random, actual_seed = utils.seed(seed)
        return actual_seed

    def reset(self) -> np.ndarray:
        p = self.params.init_position
        q = self.params.init_quaternion
        v = self.params.init_velocity
        w = self.params.init_omega
        self.state = np.concatenate([p, q, v, w])

        # 根据任务初始化目标
        if self.task == 'pose_control':
            self.target_pos = self.params.init_position.copy()
            self.target_quat = self.params.init_quaternion.copy()
        else:
            # 在此定义路径，例如直线或Bézier
            self.path = None
        return self.state.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        thrusts = np.clip(action, -self.params.max_thrusts, self.params.max_thrusts)
        tau = self.allocator.allocate(thrusts)
        self.state = self.dynamics.integrate(self.state, tau)

        reward, done = self._compute_reward(self.state)
        return self.state.copy(), float(reward), bool(done), {}

    def _compute_reward(self, state: np.ndarray) -> Tuple[float, bool]:
        if self.task == 'pose_control':
            pos, quat = state[:3], state[3:7]
            dp = np.linalg.norm(pos - self.target_pos)
            dot = np.dot(quat, self.target_quat)
            dot = np.clip(dot, -1.0, 1.0)
            angle_err = 2 * np.arccos(abs(dot))
            done = (dp < 0.05) and (angle_err < 0.1)
            reward = 1.0 if done else - (dp + 0.1 * angle_err)
            return reward, done

        # 路径跟踪奖励
        else:
            pos = state[:3]
            # TODO: 根据 self.path 计算 cross-track error
            d = np.linalg.norm(pos - self.path[0]) if self.path is not None else 0.0
            # 基础惩罚
            reward = -d
            # 阈值贴轨奖励
            delta = 0.1
            if d < delta:
                reward += 0.5
            # 航向误差
            # TODO: 计算 heading error theta
            theta = 0.0
            theta_th = np.deg2rad(10)
            if abs(theta) < theta_th:
                reward += 0.2
            # 进度奖励
            v = np.linalg.norm(state[7:10])
            v_p = v * np.cos(theta)
            reward += 0.1 * v_p
            # 最大允许误差
            max_allow_error = 0.5
            done = False
            if d > max_allow_error:
                reward -= 1.0
                done = True
            return reward, done

    def render(self, mode: str = 'human'):
        pass
