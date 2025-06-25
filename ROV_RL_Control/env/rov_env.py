# env/rov_env.py
import numpy as np
import gym
from gym import spaces
from typing import Optional, Tuple, Dict, Any

from sim.parameters import ROVParameters
from sim.hydro_dynamics import HydroDynamics
from sim.thruster_allocation import ThrusterAllocator

class ROVEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        task: str = 'pose_control',
        params: Optional[ROVParameters] = None,
        render_mode: Optional[str] = None,
        disturbance_enable: bool = False
    ) -> None:
        super().__init__()
        # 扩展参数
        self.render_mode = render_mode
        self.disturbance_enable = disturbance_enable

        # 模块初始化
        self.params = params or ROVParameters()
        self.dynamics = HydroDynamics(self.params)
        self.allocator = ThrusterAllocator(self.params)
        self.np_random = np.random.RandomState()

        # 动作空间：8 推力输出
        max_thr = self.params.max_thrusts
        self.action_space = spaces.Box(
            low=-max_thr,
            high=max_thr,
            dtype=np.float32
        )
        # 观测空间：位置(3)、四元数(4)、线速(3)、角速(3)
        inf = np.finfo(np.float32).max
        obs_high = np.concatenate([
            np.full(3, inf, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.full(3, inf, dtype=np.float32),
            np.full(3, inf, dtype=np.float32)
        ])
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        # 任务类型检查
        assert task in ('pose_control', 'path_following'), f"Unsupported task: {task}"
        self.task = task

    def reset(
        self,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境，返回初始状态和 info 字典。
        """
        if seed is not None:
            self.np_random.seed(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        # 重置内部状态
        p = self.params.init_position.copy()
        q = self.params.init_quaternion.copy()
        v = self.params.init_velocity.copy()
        w = self.params.init_omega.copy()
        self.state = np.concatenate([p, q, v, w])

        # 设置目标
        if self.task == 'pose_control':
            self.target_pos = p.copy()
            self.target_quat = q.copy()
        else:
            # TODO: 初始化路径 waypoint 列表
            self.path = []

        info: Dict[str, Any] = {}
        return self.state.copy(), info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作并返回 (state, reward, done, truncated, info)
        """
        # 计算 6自由度 力/力矩
        thrusts = np.clip(action, -self.params.max_thrusts, self.params.max_thrusts)
        force_moment = self.allocator.allocate(thrusts)
        # 状态积分
        self.state = self.dynamics.integrate(self.state, force_moment)
        # 计算奖励
        reward, done, info = self._compute_reward(self.state)
        # Gym API: terminated vs truncated（这里暂不使用 truncated）
        truncated = False
        return self.state.copy(), float(reward), bool(done), truncated, info

    def _compute_reward(
        self,
        state: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        根据当前任务类型计算 reward, done, info。
        """
        info: Dict[str, Any] = {}
        if self.task == 'pose_control':
            pos, quat = state[:3], state[3:7]
            dp = np.linalg.norm(pos - self.target_pos)
            dot = np.clip(np.dot(quat, self.target_quat), -1.0, 1.0)
            ae = 2 * np.arccos(abs(dot))
            done = (dp < 0.05) and (ae < 0.1)
            reward = 1.0 if done else - (dp + 0.1 * ae)
            info.update(position_err=dp, angle_err=ae)
            return reward, done, info

        # path_following
        pos = state[:3]
        if self.path:
            dists = [np.linalg.norm(pos - wp) for wp in self.path]
            d = min(dists)
        else:
            d = 0.0
        reward = -d
        info['cross_track'] = d
        # 阈值奖励
        if d < 0.1:
            reward += 0.5
            info['threshold_bonus'] = 0.5
        # TODO: 添加航向对齐和进度奖励
        done = False
        if d > 0.5:
            reward -= 1.0
            done = True
            info['done_reason'] = 'out_of_bounds'
        return reward, done, info

    def render(self, mode: str = 'human') -> None:
        """
        可视化接口（可扩展）
        """
        pass
