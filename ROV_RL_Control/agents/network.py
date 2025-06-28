import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MLP(nn.Module):
    """
    可选 residual 的多层感知机。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        activation: nn.Module = nn.GELU(),
        use_layernorm: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layernorm else None
        self.activation = activation
        self.residual = residual
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_layernorm:
                self.norms.append(nn.LayerNorm(dims[i+1]))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for idx, layer in enumerate(self.layers):
            prev = out
            out = layer(out)
            if self.norms is not None:
                out = self.norms[idx](out)
            out = self.activation(out)
            if self.residual and out.shape == prev.shape:
                out = out + prev
        return out


class ActorNetwork(nn.Module):
    """
    连续动作空间的 Actor 网络，支持 SAC、TD3、DDPG。
    """
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, int] = (512, 256),
        log_std_min: float = -10,
        log_std_max: float = 2,
        use_layernorm: bool = True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        # 前端归一化
        self.input_norm = nn.LayerNorm(observation_dim)
        # shared backbone
        self.backbone = MLP(
            input_dim=observation_dim,
            hidden_dims=hidden_dims,
            activation=nn.GELU(),
            use_layernorm=use_layernorm,
            residual=residual,
        )
        # SAC 分支
        self.mu_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        # TD3/DDPG 分支可以只用 mu_head
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # 输出层初始化更保守
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(state)
        x = self.backbone(x)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z).sum(-1, keepdim=True)
        # 修正 log_prob
        log_prob -= (2*(z - action) - (1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)
        mean_action = torch.tanh(mu)
        return action, log_prob, mean_action


class CriticNetwork(nn.Module):
    """
    双 Q 网络，适用于 SAC、TD3。
    """
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, int] = (512, 256),
        use_layernorm: bool = True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(observation_dim + action_dim)
        # Q1
        self.q1 = MLP(
            input_dim=observation_dim+action_dim,
            hidden_dims=hidden_dims,
            activation=nn.GELU(),
            use_layernorm=use_layernorm,
            residual=residual,
        )
        self.q1_out = nn.Linear(hidden_dims[-1], 1)
        # Q2
        self.q2 = MLP(
            input_dim=observation_dim+action_dim,
            hidden_dims=hidden_dims,
            activation=nn.GELU(),
            use_layernorm=use_layernorm,
            residual=residual,
        )
        self.q2_out = nn.Linear(hidden_dims[-1], 1)
        # 保守初始化
        nn.init.orthogonal_(self.q1_out.weight, gain=0.1)
        nn.init.zeros_(self.q1_out.bias)
        nn.init.orthogonal_(self.q2_out.weight, gain=0.1)
        nn.init.zeros_(self.q2_out.bias)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xu = torch.cat([state, action], dim=-1)
        xu = self.input_norm(xu)
        q1 = self.q1(xu)
        q1 = self.q1_out(q1)
        q2 = self.q2(xu)
        q2 = self.q2_out(q2)
        return q1, q2


class ValueNetwork(nn.Module):
    """
    可选 V 网络，用于原始 SAC。
    """
    def __init__(
        self,
        observation_dim: int,
        hidden_dims: Tuple[int, int] = (512, 256),
        use_layernorm: bool = True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(observation_dim)
        self.backbone = MLP(
            input_dim=observation_dim,
            hidden_dims=hidden_dims,
            activation=nn.GELU(),
            use_layernorm=use_layernorm,
            residual=residual,
        )
        self.out = nn.Linear(hidden_dims[-1], 1)
        nn.init.orthogonal_(self.out.weight, gain=0.1)
        nn.init.zeros_(self.out.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(state)
        x = self.backbone(x)
        return self.out(x)
