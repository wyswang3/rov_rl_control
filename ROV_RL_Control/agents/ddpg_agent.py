# 文件：rov_rl_control/agents/ddpg_agent.py

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict

from agents.network import ActorNetwork, CriticNetwork
from agents.replay_buffer import ReplayBuffer


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) 算法实现，适用于连续动作空间 (action ∈ [0,1]^action_dim)。

    核心思想：
      - 使用单一确定性策略 Actor 网络
      - 使用单一 Critic 网络估计 Q(s,a)
      - 提供探索噪声 (Gaussian) 给 Actor 输出
      - 使用目标网络 (Actor_target, Critic_target) 进行稳定更新

    主要方法：
      - select_action：根据当前 Actor 网络输出动作，可附加高斯噪声用于探索
      - store_transition：将交互 (s, a, r, s', done) 存入 ReplayBuffer
      - update：从 ReplayBuffer 采样并更新 Critic、Actor 及目标网络

    Args:
        state_dim (int): 状态维度
        action_dim (int): 动作维度
        replay_buffer_size (int): 回放缓冲区最大容量
        config (Dict): 超参数字典，可包含以下键：
            - actor_lr (float): Actor 学习率 (默认 1e-4)
            - critic_lr (float): Critic 学习率 (默认 1e-3)
            - gamma (float): 折扣因子 (默认 0.99)
            - tau (float): 软更新系数 (默认 0.005)
            - noise_std (float): 探索噪声标准差 (默认 0.1)
            - noise_clip (float): 探索噪声裁剪范围 (默认 0.2)
            - batch_size (int): 采样批大小 (默认 256)
            - device (str): 计算设备 ("cpu" 或 "cuda") (默认 "cpu")
            - hidden_dims (tuple): 网络隐藏层大小 (默认 (256,256))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay_buffer_size: int,
        config: Optional[Dict] = None
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

        cfg = config or {}
        self.actor_lr = cfg.get("actor_lr", 1e-4)
        self.critic_lr = cfg.get("critic_lr", 1e-3)
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.batch_size = cfg.get("batch_size", 256)
        self.device = torch.device(cfg.get("device", "cpu"))

        # 探索噪声相关
        self.noise_std = cfg.get("noise_std", 0.1)
        self.noise_clip = cfg.get("noise_clip", 0.2)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            max_size=replay_buffer_size,
            state_dim=state_dim,
            action_dim=action_dim
        )

        # 网络结构
        hidden_dims = cfg.get("hidden_dims", (256, 256))

        # Actor 网络及其目标网络
        self.actor = ActorNetwork(
            observation_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        for param in self.actor_target.parameters():
            param.requires_grad = False

        # Critic 网络及其目标网络
        self.critic = CriticNetwork(
            observation_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def select_action(
        self,
        state: np.ndarray,
        noise: float = 0.0
    ) -> np.ndarray:
        """
        根据当前 Actor 网络输出确定性动作，并可选添加高斯探索噪声。

        Args:
            state (np.ndarray): 状态向量，shape=(state_dim,)
            noise (float): 探索噪声标准差 (默认 0.0，表示不加噪声)

        Returns:
            action (np.ndarray): 动作向量，shape=(action_dim,), ∈ [0,1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
        with torch.no_grad():
            mu, _ = self.actor(state_tensor)  # (1, action_dim), log_std 忽略
            action = torch.tanh(mu)           # 映射到 (−1,1)
            action = (action + 1.0) / 2.0     # 映射到 [0,1]
        action = action.cpu().numpy().flatten()

        if noise > 0.0:
            action = action + np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, 0.0, 1.0)
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        将交互样本存入回放缓冲区。
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> None:
        """
        从回放缓冲区采样小批量，更新 Critic、Actor 以及目标网络。
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # ---------- 1. 从 ReplayBuffer 采样 ----------
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)           # (B, state_dim)
        actions = torch.FloatTensor(actions).to(self.device)         # (B, action_dim)
        rewards = torch.FloatTensor(rewards).to(self.device)         # (B, 1)
        next_states = torch.FloatTensor(next_states).to(self.device) # (B, state_dim)
        dones = torch.FloatTensor(dones).to(self.device)             # (B, 1)

        # ---------- 2. 计算目标 Q 值 ----------
        with torch.no_grad():
            # next_action = actor_target(next_state)
            mu_next, _ = self.actor_target(next_states)
            next_action = torch.tanh(mu_next)           # (B, action_dim)
            next_action = (next_action + 1.0) / 2.0      # 映射到 [0,1]

            # 添加平滑噪声
            noise = (torch.randn_like(next_action) * self.noise_std).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (next_action + noise).clamp(0.0, 1.0)

            # 计算目标 Q
            q1_next, q2_next = self.critic_target(next_states, next)
