# File: env/rov_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from sim.parameters import ROVParameters
from sim.hydro_dynamics import HydroDynamics
from sim.thruster_allocation import ThrusterAllocator


def quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute smallest angle between two quaternions."""
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return float(2 * np.arccos(abs(dot)))


class ROVEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    MAX_INIT_DIST = 0.3             # m
    MIN_STEPS = 5                   # minimum steps before termination
    TIME_PENALTY = 0.001            # per-step time penalty
    TRACK_SCALE = 5.0               # continuous path tracking reward scale
    WP_THRESHOLD = 0.1              # waypoint acceptance radius
    PATH_COMPLETION_REWARD = 200.0  # reward for completing all waypoints

    def __init__(
        self,
        task: str = 'pose_control',
        params: Optional[ROVParameters] = None
    ) -> None:
        super().__init__()
        assert task in ('pose_control', 'path_following'), f"Unsupported task: {task}"
        self.task = task

        self.params = params or ROVParameters()
        self.dynamics = HydroDynamics(self.params)
        self.allocator = ThrusterAllocator(self.params)
        self.rng = np.random.default_rng()

        # action: normalized commands [0,1]^8
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

        # observation: pos(3), quat(4), vel(3), omega(3), target_pos(3), target_quat(4)
        inf = np.finfo(np.float32).max
        high = np.concatenate([
            np.full(3, inf), np.ones(4), np.full(3, inf), np.full(3, inf),
            np.full(3, inf), np.ones(4)
        ]).astype(np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # internal state: pos, quat, vel, omega
        self.state = np.zeros(13, dtype=np.float32)
        # path_following placeholders
        self.path: Optional[np.ndarray] = None       # (T,3)
        self.path_quat: Optional[np.ndarray] = None  # (T,4)
        self._wp_idx: int = 0
        # pose_control targets
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.target_quat = np.array([1.,0.,0.,0.], dtype=np.float32)
        # previous errors
        self.prev_dp = 0.0
        self.prev_ae = 0.0
        self._step_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # reseed RNGs
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        # base initial state
        p = self.params.init_position.copy().astype(np.float32)
        q = self.params.init_quaternion.copy().astype(np.float32)
        v = self.params.init_velocity.copy().astype(np.float32)
        w = self.params.init_omega.copy().astype(np.float32)

        if self.task == 'path_following':
            # continuous trajectory inputs
            pos_traj = kwargs.get('path')
            quat_traj = kwargs.get('path_quat')
            assert pos_traj is not None, "path array required for path_following"
            self.path = np.asarray(pos_traj, dtype=np.float32)
            self.path_quat = np.asarray(quat_traj, dtype=np.float32) if quat_traj is not None else None
            self._wp_idx = 0
            start = self.path[0]
            self.state = np.concatenate([start, q, v, w])
            # init error
            self.prev_dp = float(np.linalg.norm(start - self.path[0]))
            self.prev_ae = quat_angle_error(q, self.path_quat[0]) if self.path_quat is not None else 0.0
        else:
            # pose control
            self.state = np.concatenate([p, q, v, w])
            if 'target_pos' in kwargs:
                self.target_pos = np.asarray(kwargs['target_pos'], dtype=np.float32)
            else:
                self.target_pos = p + self.rng.uniform(-self.MAX_INIT_DIST, self.MAX_INIT_DIST, 3).astype(np.float32)
            self.target_quat = np.asarray(kwargs.get('target_quat', q), dtype=np.float32)
            # init error
            self.prev_dp = float(np.linalg.norm(p - self.target_pos))
            self.prev_ae = quat_angle_error(q, self.target_quat)

        self._step_count = 0
        return self._get_observation(), {}

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        # state integration
        thrusts = self.allocator.map_action_to_thrust(action)
        tau = self.allocator.T.dot(thrusts)
        self.state = self.dynamics.integrate(self.state, tau)

        # update trajectory index
        if self.task == 'path_following' and self.path is not None:
            self._wp_idx = min(self._step_count, len(self.path)-1)

        reward, done, info = self._compute_reward(self.state)
        reward -= self.TIME_PENALTY
        if done and self._step_count < self.MIN_STEPS:
            done = False
        return self._get_observation(), float(reward), done, False, info

    def _get_observation(self) -> np.ndarray:
        # base state + target
        obs = self.state.copy()
        if self.task == 'pose_control':
            tgt_p, tgt_q = self.target_pos, self.target_quat
        else:
            tgt_p = self.path[self._wp_idx]
            tgt_q = self.path_quat[self._wp_idx] if self.path_quat is not None else np.zeros(4, dtype=np.float32)
        return np.concatenate([obs, tgt_p, tgt_q])

    def _compute_reward(
        self,
        state: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        pos = state[:3]
        quat = state[3:7]
        info: Dict[str, Any] = {}

        if self.task == 'pose_control':
            dp = float(np.linalg.norm(pos - self.target_pos))
            ae = quat_angle_error(quat, self.target_quat)
            delta_dp = self.prev_dp - dp
            reward = delta_dp * 10.0 - ae * 0.1
            if delta_dp < 0:
                info['penalty'] = delta_dp * 5.0
                reward += info['penalty']
            done = (dp < 0.05 and ae < 0.1)
            if done:
                reward += 100.0
                info['done_reason'] = 'reached_goal'
            info.update(position_err=dp, angle_err=ae)
            self.prev_dp, self.prev_ae = dp, ae
            return reward, done, info

        # path_following
        dp = float(np.linalg.norm(pos - self.path[self._wp_idx]))
        ae = quat_angle_error(quat, self.path_quat[self._wp_idx]) if self.path_quat is not None else 0.0
        delta_dp = self.prev_dp - dp
        reward = delta_dp * self.TRACK_SCALE - ae * 0.1
        if delta_dp < 0:
            info['penalty'] = delta_dp * 2.0
            reward += info['penalty']
        done = (self._wp_idx >= len(self.path)-1 and dp < self.WP_THRESHOLD)
        if done:
            reward += self.PATH_COMPLETION_REWARD
            info['done_reason'] = 'path_complete'
        info.update(wp_idx=self._wp_idx, position_err=dp, angle_err=ae)
        self.prev_dp = dp
        return reward, done, info
