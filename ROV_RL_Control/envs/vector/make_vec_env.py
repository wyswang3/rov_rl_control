# envs/vector/make_vec_env.py
"""
Factory for creating a vectorized ROV dynamics environment.
Supports both async and sync vectorization, plus monitoring and normalization.
"""
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from envs.rov_dyn_env import ROVDynEnv

__all__ = ["make_vec_env"]

def make_vec_env(
    cfg_env: dict,
    device: str,
    n_envs: int,
    asynchronous: bool = True
) -> gym.vector.VectorEnv:
    """
    Create a vectorized ROV dynamical environment.

    Args:
        cfg_env (dict): environment parameters, e.g. {'dt':0.02,'max_power':480.0}
        device (str): device for LSTM model ('cuda' or 'cpu')
        n_envs (int): number of parallel environments
        asynchronous (bool): use AsyncVectorEnv if True, else SyncVectorEnv

    Returns:
        VectorEnv: wrapped with VecMonitor and VecNormalize
    """
    # Closure to capture rank for seeding
    def make_fn(rank: int):
        def _init():
            env = ROVDynEnv(
                dt=cfg_env['dt'],
                max_power=cfg_env['max_power'],
                device=device
            )
            env.reset(seed=rank)
            return env
        return _init

    env_fns = [make_fn(i) for i in range(n_envs)]

    # Choose vectorization style
    if asynchronous:
        vec = AsyncVectorEnv(env_fns)
    else:
        vec = SyncVectorEnv(env_fns)

    # Monitor episode rewards and lengths
    vec = VecMonitor(vec)
    # Normalize observations (mean=0, var=1), leave rewards unnormalized
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
    return vec
