import gymnasium as gym
import numpy as np
import torch
from collections import deque
from utils.math_util import quat_mul, quat_from_omega
from models.lstm_dyn.loader import load_dynamics

class ROVDynEnv(gym.Env):
    """
    Gym 环境：基于 HybridDynamicsModel 的时序输入（window_size, 14）
    观测 (18):
      [pos(3), vel(3), eul(3), ang_vel(3), last_acc(6)]
    动作 (8): 推力功率
    """
    def __init__(self,
                 dt: float = 0.02,
                 max_power: float = 80.0,
                 device: str = "cpu",
                 window_size: int = 9):
        super().__init__()
        self.dt = dt
        self.max_power = max_power
        self.device = device
        self.window = window_size

        # Load HybridDynamicsModel
        self.model, _ = load_dynamics(device)
        self.model.eval()

        # Spaces
        self.action_space = gym.spaces.Box(0.0, max_power, (8,), np.float32)
        high = np.inf * np.ones(18, np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # State & history buffers
        self.state = np.zeros(12, dtype=np.float32)
        self.last_acc = np.zeros(6, dtype=np.float32)
        self.pw_buf  = deque(maxlen=window_size)  # each entry: 8-dim
        self.imu_buf = deque(maxlen=window_size)  # each entry: 6-dim

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Reset kinematics
        self.state.fill(0.0)
        self.state[:3] = rng.uniform(-1.0, 1.0, size=3)
        self.state[8]  = rng.uniform(0, 2*np.pi)
        self.last_acc.fill(0.0)

        # Reset buffers with zeros
        zero_pw  = np.zeros(8, dtype=np.float32)
        zero_imu = np.zeros(6, dtype=np.float32)
        self.pw_buf.clear()
        self.imu_buf.clear()
        for _ in range(self.window):
            self.pw_buf.append(zero_pw)
            self.imu_buf.append(zero_imu)

        return self._get_obs(), {}

    def step(self, action):
        # 1) Clip action & update power history
        act = np.clip(action, 0.0, self.max_power).astype(np.float32)
        self.pw_buf.append(act)

        # 2) Build LSTM input sequences
        pw_seq  = np.stack(self.pw_buf,  axis=0)  # (window,8)
        imu_seq = np.stack(self.imu_buf, axis=0)  # (window,6)
        # shape to (1,window,*) and send to device
        pw_t  = torch.from_numpy(pw_seq[None]).to(self.device)
        imu_t = torch.from_numpy(imu_seq[None]).to(self.device)

        # 3) Predict accel via HybridDynamicsModel
        with torch.no_grad():
            accel_t = self.model(pw_t, imu_t)       # (1,6)
        accel = accel_t.squeeze(0).cpu().numpy().astype(np.float32)

        # 4) Update accel history
        self.last_acc = accel
        self.imu_buf.append(accel)

        # 5) Kinematics integration
        p, v, eul, omg = np.split(self.state, [3,6,9])
        lin_acc = accel[:3]; ang_acc = accel[3:]
        v   += lin_acc * self.dt
        p   += v       * self.dt
        omg += ang_acc * self.dt

        # Orientation update
        dq   = quat_from_omega(omg, self.dt)
        q    = quat_from_omega(eul, 0.0)
        q    = quat_mul(q, dq)
        # Euler angles
        yaw   = np.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))
        roll  = np.arctan2(2*(q[3]*q[0]+q[1]*q[2]), 1-2*(q[0]**2+q[1]**2))
        pitch = np.arcsin(2*(q[3]*q[1]-q[2]*q[0]))
        eul   = np.array([roll, pitch, yaw], dtype=np.float32)

        self.state = np.hstack([p, v, eul, omg])

        # 6) Reward & done
        reward = -np.linalg.norm(p)
        done   = False
        return self._get_obs(), float(reward), done, False, {}

    def _get_obs(self):
        # concat state + last_acc
        return np.hstack([self.state, self.last_acc]).astype(np.float32)
