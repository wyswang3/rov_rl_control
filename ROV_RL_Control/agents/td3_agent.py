# 文件：rov_rl_control/agents/td3_agent.py

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Dict

from agents.network import ActorNetwork, CriticNetwork
from agents.replay_buffer import ReplayBuffer


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) 算法实现，适用于连续动作空间 (action ∈ [0,1]^action_dim)。

    核心思想：
      - 使用双 Q 网络 (Critic)，减轻过估计偏差
      - 延迟更新策略网络 (Actor) 和目标网络
      - 对目标动作加入平滑噪声 (policy smoothing) 以提高鲁棒性

    主要方法：
      - select_action：根据当前 Actor 网络输出动作
      - store_transition：将交互 (s, a, r, s', done) 存入 ReplayBuffer
      - update：从 ReplayBuffer 采样并更新 Critic、Actor 及其目标网络

    Args:
        state_dim (int): 状态维度
        action_dim (int): 动作维度
        replay_buffer_size (int): 回放缓冲区最大容量
        config (Dict): 超参数字典，可包含以下键：
            - actor_lr (float): Actor 学习率 (默认 3e-4)
            - critic_lr (float): Critic 学习率 (默认 3e-4)
            - gamma (float): 折扣因子 (默认 0.99)
            - tau (float): 软更新系数 (默认 0.005)
            - policy_noise (float): 目标动作添加噪声标准差 (默认 0.2)
            - noise_clip (float): 平滑噪声裁剪范围 (默认 0.5)
            - policy_freq (int): 延迟更新 Actor 和目标网络的频率 (默认 2)
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
        self.actor_lr = cfg.get("actor_lr", 3e-4)
        self.critic_lr = cfg.get("critic_lr", 3e-4)
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.batch_size = cfg.get("batch_size", 256)
        self.device = torch.device(cfg.get("device", "cpu"))

        # TD3 特有超参数
        self.policy_noise = cfg.get("policy_noise", 0.2)      # 目标动作噪声标准差
        self.noise_clip = cfg.get("noise_clip", 0.5)          # 噪声裁剪范围
        self.policy_freq = cfg.get("policy_freq", 2)          # 延迟更新频率

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

        # Critic 网络 (双 Q) 及其目标网络
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

        # 更新计数器，用于延迟更新
        self.total_it = 0

    def select_action(
        self,
        state: np.ndarray,
        noise: float = 0.0
    ) -> np.ndarray:
        """
        根据当前 Actor 网络输出动作，并可选添加高斯探索噪声。

        Args:
            state (np.ndarray): 状态向量，shape=(state_dim,)
            noise (float): 探索噪声标准差 (默认 0.0，表示不加噪声)

        Returns:
            action (np.ndarray): 动作向量，shape=(action_dim,), ∈ [0,1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
        with torch.no_grad():
            mu, _ = self.actor(state_tensor)  # (1, action_dim), log_std 忽略
            action = torch.tanh(mu)           # 映射到 (-1,1)
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
        从回放缓冲区采样小批量，更新 Critic、Actor 以及目标网络。Actor 和目标网络延迟更新。
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        self.total_it += 1

        # ---------- 1. 从 ReplayBuffer 采样 ----------
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)           # (B, state_dim)
        actions = torch.FloatTensor(actions).to(self.device)         # (B, action_dim)
        rewards = torch.FloatTensor(rewards).to(self.device)         # (B, 1)
        next_states = torch.FloatTensor(next_states).to(self.device) # (B, state_dim)
        dones = torch.FloatTensor(dones).to(self.device)             # (B, 1)

        # ---------- 2. 计算目标动作，并加入平滑噪声 ----------
        with torch.no_grad():
            # next_action = actor_target(next_state)
            mu_next, _ = self.actor_target(next_states)
            next_action = torch.tanh(mu_next)           # (-1,1)
            next_action = (next_action + 1.0) / 2.0      # [0,1]

            # 添加噪声并裁剪
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = next_action + noise
            next_action = next_action.clamp(0.0, 1.0)

            # 计算目标 Q 值
            q1_next, q2_next = self.critic_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1.0 - dones) * q_next  # (B,1)

        # ---------- 3. 更新 Critic 网络 ----------
        current_q1, current_q2 = self.critic(states, actions)  # (B,1)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------- 4. 延迟更新 Actor 和目标网络 ----------
        if self.total_it % self.policy_freq == 0:
            # a) 更新 Actor 网络：最大化 Q(s, π(s)) -> 最小化 -Q
            mu, _ = self.actor(states)
            action_pred = torch.tanh(mu)
            action_pred = (action_pred + 1.0) / 2.0
            actor_loss = -self.critic.q1_fc1(torch.cat([states, action_pred], dim=-1))  # 不直接调用 forward，需要提取 Q1
            # 更直接地通过 critic.forward 获取 q1:
            q1_pred, _ = self.critic(states, action_pred)
            actor_loss = -q1_pred.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # b) 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, save_path: str) -> None:
        """
        保存模型权重：
          - actor, critic
        """
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        torch.save(checkpoint, save_path)

    def load(self, load_path: str) -> None:
        """
        加载模型权重，并同步目标网络。
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
