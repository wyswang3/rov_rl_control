# 文件：rov_rl_control/agents/sac_agent.py

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict
import torch.nn.functional as F
from agents.network import ActorNetwork, CriticNetwork, ValueNetwork
from agents.replay_buffer import ReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic (SAC) 算法实现，适用于连续动作空间（action ∈ [0,1]^action_dim）。
    使用双 Q 网络、策略网络（带自动熵调节）和状态价值网络（以及目标价值网络）。

    主要功能：
      - select_action：根据当前策略，从状态中采样动作
      - store_transition：将交互 (s, a, r, s', done) 存入 ReplayBuffer
      - update：从 ReplayBuffer 采样小批量，更新网络参数
      - save / load：保存与加载模型权重

    Args:
        state_dim (int): 状态维度
        action_dim (int): 动作维度
        replay_buffer_size (int): 回放缓冲区最大容量
        config (Dict): 超参数字典，包含以下可选键：
            - lr (float): 学习率 (默认 3e-4)
            - gamma (float): 折扣因子 (默认 0.99)
            - tau (float): 软更新系数 (默认 0.005)
            - alpha (float): 初始熵系数 (若自动调节则为初始值) (默认 0.2)
            - automatic_entropy_tuning (bool): 是否自动调节熵系数 (默认 True)
            - target_entropy (float): 目标熵 (默认 -action_dim)
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
        # Hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim

        cfg = config or {}
        self.lr = cfg.get("lr", 3e-4)
        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)
        self.batch_size = cfg.get("batch_size", 256)
        self.device = torch.device(cfg.get("device", "cpu"))

        # Automatic entropy tuning
        self.automatic_entropy_tuning = cfg.get("automatic_entropy_tuning", True)
        if self.automatic_entropy_tuning:
            # 初始 alpha
            self.target_entropy = cfg.get("target_entropy", -action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = cfg.get("alpha", 0.2)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            max_size=replay_buffer_size,
            state_dim=state_dim,
            action_dim=action_dim
        )

        # Networks
        hidden_dims = cfg.get("hidden_dims", (256, 256))
        # Actor (输出 mu, log_std)
        self.actor = ActorNetwork(
            observation_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            log_std_min=cfg.get("log_std_min", -20),
            log_std_max=cfg.get("log_std_max", 2),
        ).to(self.device)

        # Critic (双 Q 网络)
        self.critic = CriticNetwork(
            observation_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        # 复制一个目标 Critic，用于延迟更新
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Value 网络及其目标网络 (可选：使用 V 网络)
        self.use_value_net = cfg.get("use_value_net", True)
        if self.use_value_net:
            self.value_net = ValueNetwork(
                observation_dim=state_dim,
                hidden_dims=hidden_dims
            ).to(self.device)
            self.value_target = copy.deepcopy(self.value_net).to(self.device)
            for param in self.value_target.parameters():
                param.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        if self.use_value_net:
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False
    ) -> np.ndarray:
        """
        根据当前策略从状态中采样动作。
        Args:
            state (np.ndarray): 状态向量，shape = (state_dim,)
            evaluate (bool): 是否评估模式（使用 mean 动作，无噪声）
        Returns:
            action (np.ndarray): 动作向量，shape = (action_dim,), ∈ [0,1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
        with torch.no_grad():
            mu, log_std = self.actor(state_tensor)
            if evaluate:
                # 直接使用均值，并映射到 [0,1]
                mean = torch.tanh(mu)
                action = (mean + 1.0) / 2.0
                action = action.clamp(0.0, 1.0)
                return action.cpu().numpy().flatten()
            else:
                # 使用采样
                z = mu + log_std.exp() * torch.randn_like(mu)
                action_tanh = torch.tanh(z)
                action = (action_tanh + 1.0) / 2.0  # 映射到 [0,1]
                return action.cpu().numpy().flatten()

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
        从回放缓冲区采样小批量，更新 actor、critic、value 及熵系数（如果自动调节）。
        该方法在训练循环中被多次调用，确保少量梯度更新。
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

        # ---------- 2. 计算目标 Q-value ----------
        with torch.no_grad():
            # a) 在 next_states 上采样 next_action 及 log_prob
            next_mu, next_log_std = self.actor(next_states)
            next_std = next_log_std.exp()
            normal = torch.distributions.Normal(next_mu, next_std)
            z = normal.rsample()
            next_action_tanh = torch.tanh(z)
            next_action = (next_action_tanh + 1.0) / 2.0  # 映射到 [0,1]

            log_prob = normal.log_prob(z)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            # 修正 log_prob
            log_prob -= (2 * (z - next_action_tanh) - (1 - next_action_tanh.pow(2))).sum(dim=-1, keepdim=True)

            # b) 计算 Q_target：Q1'(s',a'), Q2'(s',a')
            q1_next, q2_next = self.critic_target(next_states, next_action)
            q_next_min = torch.min(q1_next, q2_next)

            # c) 如果使用 V 网络版本（原始 SAC），则 V_target = q_next_min - alpha * log_prob
            if self.use_value_net:
                v_target = q_next_min - self._get_alpha() * log_prob
                # 计算 y = reward + gamma * (1 - done) * V_target
                y = rewards + self.gamma * (1.0 - dones) * v_target
            else:
                # 如果不使用 V 网络，则直接通过 Q-Backup:
                y = rewards + self.gamma * (1.0 - dones) * (q_next_min - self._get_alpha() * log_prob)

        # ---------- 3. 更新 Critic 网络 ----------
        q1, q2 = self.critic(states, actions)  # (B,1), (B,1)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------- 4. 更新 Value 网络（若启用） ----------
        if self.use_value_net:
            # V(s) = value_net(states)
            v = self.value_net(states)  # (B,1)
            # 对应 Q 网络下的 A-value: Q(s,a) - alpha * log_prob(a|s)
            with torch.no_grad():
                mu, log_std = self.actor(states)
                std = log_std.exp()
                normal2 = torch.distributions.Normal(mu, std)
                z2 = normal2.rsample()
                action_tanh2 = torch.tanh(z2)
                action2 = (action_tanh2 + 1.0) / 2.0
                log_prob2 = normal2.log_prob(z2).sum(dim=-1, keepdim=True)
                log_prob2 -= (2 * (z2 - action_tanh2) - (1 - action_tanh2.pow(2))).sum(dim=-1, keepdim=True)
                q1_pi, q2_pi = self.critic(states, action2)
                q_pi = torch.min(q1_pi, q2_pi)
                # A = Q(s,a) - alpha * log_prob
                a_value = q_pi - self._get_alpha() * log_prob2

            # V 损失: (V(s) - A)^2
            value_loss = F.mse_loss(v, a_value)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # ---------- 5. 更新 Policy (Actor) 网络 ----------
        mu, log_std = self.actor(states)
        std = log_std.exp()
        normal3 = torch.distributions.Normal(mu, std)
        z3 = normal3.rsample()
        action_tanh3 = torch.tanh(z3)
        action3 = (action_tanh3 + 1.0) / 2.0
        log_prob3 = normal3.log_prob(z3).sum(dim=-1, keepdim=True)
        log_prob3 -= (2 * (z3 - action_tanh3) - (1 - action_tanh3.pow(2))).sum(dim=-1, keepdim=True)

        # 计算 Q(s, π(s))
        q1_pi, q2_pi = self.critic(states, action3)
        q_pi = torch.min(q1_pi, q2_pi)

        # 策略损失: E[ α * log_prob - Q ]
        actor_loss = (self._get_alpha() * log_prob3 - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------- 6. 自动调节 α（若启用） ----------
        if self.automatic_entropy_tuning:
            # 目标：E[−log_prob] → target_entropy
            alpha_loss = -(self.log_alpha * (log_prob3 + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # ---------- 7. 软更新目标网络 ----------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.use_value_net:
            for param, target_param in zip(self.value_net.parameters(), self.value_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _get_alpha(self) -> float:
        """
        获取当前熵系数 α。
        如果使用自动调节，则每次更新后从 log_alpha 中取 exp；否则返回固定 alpha。
        """
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        else:
            return self.alpha

    def save(self, save_path: str) -> None:
        """
        保存模型权重：
          - actor, critic, (value), (log_alpha)
        """
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        if self.use_value_net:
            checkpoint["value_net"] = self.value_net.state_dict()
        if self.automatic_entropy_tuning:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu().numpy()
        torch.save(checkpoint, save_path)

    def load(self, load_path: str) -> None:
        """
        加载模型权重，并同步 target 网络。
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target = copy.deepcopy(self.critic)
        if self.use_value_net:
            self.value_net.load_state_dict(checkpoint["value_net"])
            self.value_target = copy.deepcopy(self.value_net)
        if self.automatic_entropy_tuning and "log_alpha" in checkpoint:
            log_alpha_val = checkpoint["log_alpha"]
            self.log_alpha = torch.tensor(log_alpha_val, requires_grad=True, device=self.device)
