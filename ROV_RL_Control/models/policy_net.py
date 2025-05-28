# models/policy_net.py
"""
Custom policy network settings for Stable-Baselines3.
Includes a feature extractor mapping observations to a latent space,
and policy_kwargs to configure the Actor-Critic network.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ROVFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that maps the 18-dimensional observation to a
    lower-dimensional embedding for the policy and value networks.

    Args:
        observation_space (gym.Space): Input observation space
        features_dim (int): Dimension of the features extracted
    """
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_obs = observation_space.shape[0]
        # Define a small MLP
        self.net = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, n_obs)
        return self.net(observations)

# Policy kwargs to pass when initializing SB3 model
policy_kwargs = {
    'features_extractor_class': ROVFeaturesExtractor,
    'features_extractor_kwargs': {
        'features_dim': 256,
    },
    'net_arch': [
        {
            'pi': [128, 128],  # actor network layers
            'vf': [128, 128],  # value network layers
        }
    ],
    'activation_fn': nn.ReLU,
}
