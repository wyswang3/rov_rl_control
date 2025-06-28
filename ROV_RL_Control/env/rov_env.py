# File: env/rov_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from sim.parameters import ROVParameters
from sim.hydro_dynamics import HydroDynamics
from sim.thruster_allocation import ThrusterAllocator


def quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute smallest angle between two quaternions."""
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return float(2 * np.arccos(abs(dot)))


class ROVEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    # termination & reward constants
    MAX_INIT_DIST = 0.3
    MIN_STEPS = 5
    TIME_PENALTY = 1e-4
    TRACK_SCALE = 20.0
    WP_THRESHOLD = 0.1
    PATH_COMPLETION_REWARD = 200.0

    def __init__(
        self,
        task: str = 'pose_control',
        params: Optional[ROVParameters] = None
    ) -> None:
        super().__init__()
        assert task in ('pose_control', 'path_following'), f"Unsupported task: {task}"
        self.task = task

        # physics & action
        self.params = params or ROVParameters()
        self.dynamics = HydroDynamics(self.params)
        self.allocator = ThrusterAllocator(self.params)
        self.rng = np.random.default_rng()
        self.action_space = spaces.Box(0.0, 1.0, (8,), np.float32)

        # observation: [pos3,quat4,vel3,omega3,target_pos3,target_quat4]
        self.obs_dim = 13 + 3 + 4
        high = np.inf * np.ones(self.obs_dim, np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # internal state
        self.state = np.zeros(13, np.float32)
        self.path = None       # type: Optional[np.ndarray]
        self.path_quat = None  # type: Optional[np.ndarray]
        self._wp_idx = 0

        # pose targets
        self.target_pos = np.zeros(3, np.float32)
        self.target_quat = np.array([1,0,0,0], np.float32)

        # shaping helpers
        self.prev_dp = 0.0
        self.prev_ae = 0.0
        self._step_count = 0

        # running stats for obs normalization
        self.obs_count = 0
        self.obs_mean = np.zeros(self.obs_dim, np.float64)
        self.obs_M2   = np.zeros(self.obs_dim, np.float64)

    def _update_running_stats(self, obs: np.ndarray):
        """Welford's algorithm update."""
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_M2 += delta * delta2

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize obs using running stats; leave quats raw."""
        if self.obs_count < 2:
            return obs.astype(np.float32)
        var = self.obs_M2 / (self.obs_count - 1)
        std = np.sqrt(var + 1e-8)
        norm = (obs - self.obs_mean) / std
        # keep quaternion dims raw
        for i in list(range(3,7)) + list(range(16,20)):
            norm[i] = obs[i]
        return np.clip(norm, -5.0, 5.0).astype(np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # reseed
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        # base state
        p = self.params.init_position.astype(np.float32)
        q = self.params.init_quaternion.astype(np.float32)
        v = self.params.init_velocity.astype(np.float32)
        w = self.params.init_omega.astype(np.float32)

        if self.task == 'path_following':
            pos_traj = kwargs.get('path')
            assert pos_traj is not None, "path array required for path_following"
            self.path = np.asarray(pos_traj, dtype=np.float32)
            self.path_quat = None
            self._wp_idx = 0
            start = self.path[0]
            self.state = np.concatenate([start, q, v, w])
            self.prev_dp = np.linalg.norm(start - self.path[0])
            self.prev_ae = 0.0
        else:
            self.state = np.concatenate([p, q, v, w])
            if 'target_pos' in kwargs:
                self.target_pos = np.asarray(kwargs['target_pos'], np.float32)
            else:
                cand = p + self.rng.uniform(-self.MAX_INIT_DIST, self.MAX_INIT_DIST, 3)
                self.target_pos = cand.astype(np.float32)
            self.target_quat = np.asarray(kwargs.get('target_quat', q), np.float32)
            self.prev_dp = np.linalg.norm(p - self.target_pos)
            self.prev_ae = quat_angle_error(q, self.target_quat)

        self._step_count = 0

        raw = self._get_observation()
        self._update_running_stats(raw)
        return self._normalize_obs(raw), {'raw_observation': raw}

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1

        # physics
        thrusts = self.allocator.map_action_to_thrust(action)
        tau = self.allocator.T.dot(thrusts)
        self.state = self.dynamics.integrate(self.state, tau)

        # clamp waypoint index
        if self.task == 'path_following' and self.path is not None:
            self._wp_idx = min(self._step_count, len(self.path)-1)

        # compute reward & done
        reward, done, info = self._compute_reward(self.state)
        reward -= self.TIME_PENALTY
        # control effort penalty
        info['ctrl_penalty'] = 0.1 * np.linalg.norm(action)**2
        reward -= info['ctrl_penalty']

        if done and self._step_count < self.MIN_STEPS:
            done = False

        # obs
        raw = self._get_observation()
        self._update_running_stats(raw)
        norm = self._normalize_obs(raw)
        info['raw_observation'] = raw
        return norm, float(reward), done, False, info

    def _get_observation(self) -> np.ndarray:
        obs = self.state.copy()
        if self.task == 'pose_control':
            tgt_p, tgt_q = self.target_pos, self.target_quat
        else:
            idx = min(self._wp_idx, len(self.path)-1)
            tgt_p = self.path[idx]
            tgt_q = np.zeros(4, dtype=np.float32)
        return np.concatenate([obs, tgt_p, tgt_q])

    def _compute_reward(
        self,
        state: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}

        pos, quat = state[:3], state[3:7]

        if self.task == 'pose_control':
            dp = float(np.linalg.norm(pos - self.target_pos))
            ae = quat_angle_error(quat, self.target_quat)

            # dead-zone shaping
            DEAD = 0.2
            reward = -max(dp - DEAD, 0.0) * 0.05

            # progress shaping
            delta = self.prev_dp - dp
            reward += (delta * 5.0) if delta>0 else (delta * 2.0)

            # small attitude bonus
            if ae < 0.2:
                reward += 1.0

            done = False
            if dp < 0.05 and ae < 0.1:
                reward += 50.0
                done = True
                info['done_reason'] = 'reached_goal'

            info.update(position_err=dp, delta_dp=delta, angle_err=ae)
            self.prev_dp, self.prev_ae = dp, ae
            return reward, done, info

        # path_following
        dp = float(np.linalg.norm(pos - self.path[self._wp_idx]))
        ae = 0.0
        delta = self.prev_dp - dp

        # dead-zone shaping
        DEAD = 0.2
        reward = -max(dp - DEAD, 0.0) * 0.05

        # progress shaping
        reward += (delta * self.TRACK_SCALE) if delta>0 else (delta * (self.TRACK_SCALE/5))

        # waypoint bonus
        done = False
        if dp < self.WP_THRESHOLD:
            reward += 10.0
            self._wp_idx += 1
            if self._wp_idx >= len(self.path):
                reward += self.PATH_COMPLETION_REWARD
                done = True
                info['done_reason'] = 'path_complete'

        info.update(wp_idx=self._wp_idx, position_err=dp, delta_dp=delta)
        self.prev_dp = dp
        return reward, done, info
