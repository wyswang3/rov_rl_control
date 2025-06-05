# 文件：rov_rl_control/agents/network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ActorNetwork(nn.Module):
    """
    连续动作空间下的 Actor 网络（适用于 SAC、TD3、DDPG 等算法）。
    输入：状态向量（observation_dim,）
    输出：若干种设计方式：
      - 对于确定性策略（TD3、DDPG）：输出 8 维动作（未经激活时为实数，需要后续映射到 [0, 1] 或 [−1, 1]）
      - 对于随机策略（SAC）：输出动作均值 mu (8,) 和 log_std (8,)
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, int] = (256, 256),
        log_std_min: float = -20,
        log_std_max: float = 2,
    ) -> None:
        """
        Args:
            observation_dim (int): 状态维度
            action_dim (int): 动作维度（本项目为 8）
            hidden_dims (tuple): 两层隐藏层大小，默认为 (256, 256)
            log_std_min (float): 对 SAC 中 log_std 下限剪裁
            log_std_max (float): 对 SAC 中 log_std 上限剪裁
        """
        super().__init__()
        self.obs_dim = observation_dim
        self.act_dim = action_dim

        # 第一层
        self.fc1 = nn.Linear(observation_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        # 对于 SAC: 同时输出 mu 和 log_std
        self.mu_head = nn.Linear(hidden_dims[1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[1], action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 对于确定性策略（TD3/DDPG），可以直接用 mu_head 输出动作

        # 权重初始化（可选，增加稳定性）
        self._init_weights()

    def _init_weights(self) -> None:
        """
        初始化全连接层权重，使用 Xavier 均匀分布。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对于 SAC，输出：mu (batch, action_dim) 和 log_std (batch, action_dim)
        对于确定性策略，可只使用 mu：
          在 TD3 中可忽略 log_std，直接用 mu 作为动作
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对于 SAC，从高斯分布中采样动作，并返回：
          - action (batch, action_dim)：已经通过 tanh 映射到 (-1,1)，后续再映射到 [0,1]
          - log_prob (batch, 1)：对数概率
          - mean_action (batch, action_dim)：未加噪声的 tanh(mu)（可用于评估）

        注意：这里假设动作空间初步在 (−1,1)，若需要 [0,1]，在外部再行 (action+1)/2 转换。
        """
        mu, log_std = self.forward(state)  # (batch, act_dim)
        std = log_std.exp()

        # 1) 采样高斯噪声
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()  # 重参数采样 (batch, act_dim)

        # 2) 计算 log_prob
        log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)  # (batch, 1)
        # 3) 通过 tanh 将动作值映射到 (-1,1)
        action = torch.tanh(z)
        # 4) 修正 log_prob：见 SAC 原文 Appendix C
        #    log_prob = log_prob - sum(log(1 - tanh(z)^2) + ε)
        log_prob -= (2 * (z - action) - (1 - action.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        # 5) 计算 mean_action
        mean_action = torch.tanh(mu)

        return action, log_prob, mean_action


class CriticNetwork(nn.Module):
    """
    Q 网络（双网络结构，适用于 SAC、TD3）：
      共享前两层或各自独立，两输入：状态 + 动作 → 输出 Q(s, a)
      对于双 Q，需要初始化两个并行网络。
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, int] = (256, 256),
    ) -> None:
        """
        Args:
            observation_dim (int): 状态维度
            action_dim (int): 动作维度
            hidden_dims (tuple): 隐藏层大小，默认为 (256, 256)
        """
        super().__init__()
        input_dim = observation_dim + action_dim

        # Q1 网络
        self.q1_fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.q1_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.q1_out = nn.Linear(hidden_dims[1], 1)

        # Q2 网络
        self.q2_fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.q2_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.q2_out = nn.Linear(hidden_dims[1], 1)

        # 权重初始化
        self._init_weights()

    def _init_weights(self) -> None:
        """
        初始化所有线性层的权重。
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，输入：
          - state: (batch, observation_dim)
          - action: (batch, action_dim)
        返回：
          - q1: (batch, 1)
          - q2: (batch, 1)
        """
        xu = torch.cat([state, action], dim=-1)  # (batch, obs+act)

        # Q1
        x1 = F.relu(self.q1_fc1(xu))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)

        # Q2
        x2 = F.relu(self.q2_fc1(xu))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)

        return q1, q2


class ValueNetwork(nn.Module):
    """
    仅在某些算法（如原始 SAC）需要 Value 网络时使用。
    输入：状态 → 输出：V(s)
    """

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: Tuple[int, int] = (256, 256),
    ) -> None:
        """
        Args:
            observation_dim (int): 状态维度
            hidden_dims (tuple): 隐藏层大小
        """
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.out = nn.Linear(hidden_dims[1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        输入：
          - state: (batch, observation_dim)
        返回：
          - value: (batch, 1)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.out(x)
        return value
