import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict

from agents.network import ActorNetwork, CriticNetwork
from agents.replay_buffer import ReplayBuffer


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) with train/eval modes and debug support.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay_buffer_size: int,
        config: Optional[Dict] = None,
        debug: bool = False
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mode = 'train'
        self.debug = debug

        cfg = config or {}
        self.actor_lr = cfg.get('actor_lr', 3e-4)
        self.critic_lr = cfg.get('critic_lr', 3e-4)
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.005)
        self.batch_size = cfg.get('batch_size', 256)
        self.device = torch.device(cfg.get('device', 'cpu'))

        # TD3-specific
        self.policy_noise = cfg.get('policy_noise', 0.2)
        self.noise_clip = cfg.get('noise_clip', 0.5)
        self.policy_freq = cfg.get('policy_freq', 2)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=replay_buffer_size,
            state_dim=state_dim,
            action_dim=action_dim
        )

        # Networks
        hidden_dims = cfg.get('hidden_dims', (256, 256))
        self.actor = ActorNetwork(
            observation_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic = CriticNetwork(
            observation_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.total_it = 0

    def set_mode(self, mode: str) -> None:
        assert mode in ('train', 'eval')
        self.mode = mode
        if mode == 'train':
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()

    def select_action(
        self,
        state: np.ndarray,
        noise: float = 0.0
    ) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _ = self.actor(state_t)
            action = torch.tanh(mu)
            action = (action + 1.0) / 2.0
        action = action.cpu().numpy().flatten()
        if self.mode == 'train' and noise > 0.0:
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
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> None:
        if self.mode != 'train' or len(self.replay_buffer) < self.batch_size:
            return

        self.total_it += 1
        # Sample batch
        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(s).to(self.device)
        actions = torch.FloatTensor(a).to(self.device)
        rewards = torch.FloatTensor(r).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(ns).to(self.device)
        dones = torch.FloatTensor(d).unsqueeze(-1).to(self.device)

        # Compute target actions with smoothing noise
        with torch.no_grad():
            mu_ns, _ = self.actor_target(next_states)
            next_action = torch.tanh(mu_ns)
            next_action = (next_action + 1.0) / 2.0
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(0.0, 1.0)
            # Target Q
            q1_ns, q2_ns = self.critic_target(next_states, next_action)
            q_ns = torch.min(q1_ns, q2_ns)
            target_q = rewards + self.gamma * (1 - dones) * q_ns

        # Critic update
        q1_s, q2_s = self.critic(states, actions)
        critic_loss = F.mse_loss(q1_s, target_q) + F.mse_loss(q2_s, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed actor update
        if self.total_it % self.policy_freq == 0:
            # Actor loss
            mu_s, _ = self.actor(states)
            action_pred = torch.tanh(mu_s)
            action_pred = (action_pred + 1.0) / 2.0
            q1_pred, _ = self.critic(states, action_pred)
            actor_loss = -q1_pred.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # Soft update targets
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        if self.debug:
            print(f"[TD3Agent] critic_loss={critic_loss.item():.4f}, actor_update={'Yes' if self.total_it%self.policy_freq==0 else 'No'}")

    def save(self, path: str) -> None:
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
