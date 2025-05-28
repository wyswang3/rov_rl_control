import pytest
import numpy as np
import torch
from envs.rov_dyn_env import ROVDynEnv


def test_env_initialization_and_shapes():
    """
    Test that the environment initializes correctly and returns observations
    of the expected shape without NaNs or infinities.
    """
    # Test on CPU; if GPU is available, also test on CUDA
    for device in ['cpu', 'cuda']:
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping GPU test")

        env = ROVDynEnv(dt=0.02, max_power=480.0, device=device)
        obs, info = env.reset(seed=0)
        # Observation should be 18-dimensional vector
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (18,)
        # All values finite
        assert np.all(np.isfinite(obs))

        # Action space sample
        action = env.action_space.sample()
        assert action.shape == (8,)
        assert np.all(action >= 0) and np.all(action <= 480.0)

        # Step multiple times
        for _ in range(100):
            next_obs, reward, done, truncated, info = env.step(action)
            # Check observation
            assert isinstance(next_obs, np.ndarray)
            assert next_obs.shape == (18,)
            assert np.all(np.isfinite(next_obs))
            # Check reward
            assert isinstance(reward, float)
            assert np.isfinite(reward)
            # Check done/truncated flags
            assert isinstance(done, bool)
            assert isinstance(truncated, bool)

        env.close()
