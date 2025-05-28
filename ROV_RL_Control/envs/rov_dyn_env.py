import gymnasium as gym
import numpy as np
import torch
from utils.math_util import quat_mul, quat_from_omega
from models.lstm_dyn.loader import load_dynamics

env_metadata = {"render_modes": []}

class ROVDynEnv(gym.Env):
    """
    Gym environment for ROV dynamics based on a pretrained LSTM model.

    Observation (18-dim):
      [pos_x, pos_y, pos_z,
       vel_x, vel_y, vel_z,
       roll, pitch, yaw,
       ang_x, ang_y, ang_z,
       acc_x, acc_y, acc_z, ang_acc_x, ang_acc_y, ang_acc_z]
    Action (8-dim):  thruster power values in [0, max_power]
    """
    metadata = env_metadata

    def __init__(self,
                 dt: float = 0.02,
                 max_power: float = 480.0,
                 device: str = "cuda"):
        super().__init__()
        # Simulation parameters
        self.dt = dt
        self.max_power = max_power
        self.device = device

        # Load pretrained LSTM dynamics model
        self.model, self.init_hidden_fn = load_dynamics(device)
        self.hidden = None

        # Action space: 8 thrusters
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=self.max_power,
            shape=(8,),
            dtype=np.float32
        )
        # Observation: 12 state + 6 last acceleration
        obs_dim = 18
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        # Internal state buffers
        self.state = np.zeros(12, dtype=np.float32)   # [pos(3), vel(3), eul(3), ang_vel(3)]
        self.last_acc = np.zeros(6, dtype=np.float32) # [lin_acc(3), ang_acc(3)]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        # Random position in [-1,1] and random yaw, others zero
        self.state.fill(0.0)
        self.state[0:3] = rng.uniform(-1.0, 1.0, size=3)
        self.state[8] = rng.uniform(0.0, 2*np.pi)  # yaw
        self.last_acc.fill(0.0)
        # Initialize hidden state of LSTM
        self.hidden = self.init_hidden_fn(batch=1)
        return self._get_obs(), {}

    def step(self, action):
        # Clip and prepare input
        act = np.clip(action, 0.0, self.max_power).astype(np.float32)
        inp = np.hstack([act, self.last_acc]).reshape(1, -1)
        inp_t = torch.from_numpy(inp).to(self.device)

        # Predict next acceleration via LSTM
        with torch.no_grad():
            acc_pred, self.hidden = self.model(inp_t, self.hidden)
        acc = acc_pred.cpu().numpy().reshape(-1).astype(np.float32)
        self.last_acc = acc.copy()

        # Unpack state
        p, v, eul, omg = np.split(self.state, [3, 6, 9])
        lin_acc = acc[0:3]
        ang_acc = acc[3:6]
        # Euler integration for velocities and positions
        v += lin_acc * self.dt
        p += v * self.dt
        omg += ang_acc * self.dt
        # Quaternion update for orientation
        dq = quat_from_omega(omg, self.dt)
        # Convert current euler to quat
        # approximate: use small-angle for simplicity
        q = quat_from_omega(eul, 0.0)  # treat eul as previous angular increment
        q = quat_mul(q, dq)
        # Back to Euler
        # yaw from quaternion
        yaw = np.arctan2(2*(q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2))
        # roll, pitch approximate small-angle
        roll = np.arctan2(2*(q[3]*q[0] + q[1]*q[2]), 1 - 2*(q[0]**2 + q[1]**2))
        pitch = np.arcsin(2*(q[3]*q[1] - q[2]*q[0]))
        eul = np.array([roll, pitch, yaw], dtype=np.float32)

        # Update state
        self.state = np.hstack([p, v, eul, omg])

        # Compute reward (distance to origin)
        reward = -np.linalg.norm(p)
        done = False
        return self._get_obs(), float(reward), done, False, {}

    def _get_obs(self):
        return np.hstack([self.state, self.last_acc]).astype(np.float32)
