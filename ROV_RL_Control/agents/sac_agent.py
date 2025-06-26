import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict

from agents.network import ActorNetwork, CriticNetwork, ValueNetwork
from agents.replay_buffer import ReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic (SAC) implementation with train/eval modes and debug support.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay_buffer_size: int,
        config: Optional[Dict] = None,
        debug: bool = False
    ) -> None:
        # Store dims and mode
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mode = 'train'
        self.debug = debug

        cfg = config or {}
        self.lr = cfg.get('lr', 3e-4)
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.005)
        self.batch_size = cfg.get('batch_size', 256)
        self.device = torch.device(cfg.get('device', 'cpu'))

        # Entropy tuning
        self.automatic_entropy_tuning = cfg.get('automatic_entropy_tuning', True)
        if self.automatic_entropy_tuning:
            self.target_entropy = cfg.get('target_entropy', -action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = cfg.get('alpha', 0.2)

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
            hidden_dims=hidden_dims,
            log_std_min=cfg.get('log_std_min', -20),
            log_std_max=cfg.get('log_std_max', 2)
        ).to(self.device)

        self.critic = CriticNetwork(
            observation_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.use_value_net = cfg.get('use_value_net', True)
        if self.use_value_net:
            self.value_net = ValueNetwork(
                observation_dim=state_dim,
                hidden_dims=hidden_dims
            ).to(self.device)
            self.value_target = copy.deepcopy(self.value_net).to(self.device)
            for p in self.value_target.parameters():
                p.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        if self.use_value_net:
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

    def set_mode(self, mode: str) -> None:
        """Set agent to 'train' or 'eval' mode, toggling network behaviors."""
        assert mode in ('train', 'eval')
        self.mode = mode
        if mode == 'train':
            self.actor.train()
            self.critic.train()
            if self.use_value_net:
                self.value_net.train()
        else:
            self.actor.eval()
            self.critic.eval()
            if self.use_value_net:
                self.value_net.eval()

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False
    ) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, log_std = self.actor(state_t)
            if evaluate:
                action = torch.tanh(mu)
            else:
                std = log_std.exp()
                z = mu + std * torch.randn_like(mu)
                action = torch.tanh(z)
            # map [-1,1] to [0,1]
            action = (action + 1.0) / 2.0
            return action.clamp(0.0, 1.0).cpu().numpy().flatten()

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

        # Sample
        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(s).to(self.device)
        actions = torch.FloatTensor(a).to(self.device)
        rewards = torch.FloatTensor(r).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(ns).to(self.device)
        dones = torch.FloatTensor(d).unsqueeze(-1).to(self.device)

        # Compute targets
        with torch.no_grad():
            # Next actions and log_probs
            mu_ns, log_std_ns = self.actor(next_states)
            std_ns = log_std_ns.exp()
            dist = torch.distributions.Normal(mu_ns, std_ns)
            z = dist.rsample()
            a_ns = torch.tanh(z)
            logp_ns = dist.log_prob(z).sum(-1, keepdim=True)
            # Correction
            logp_ns -= (2 * (z - a_ns) - (1 - a_ns.pow(2))).sum(-1, keepdim=True)

            q1_ns, q2_ns = self.critic_target(next_states, a_ns)
            q_ns = torch.min(q1_ns, q2_ns)
            alpha = self._get_alpha()

            if self.use_value_net:
                v_tgt = q_ns - alpha * logp_ns
                y_q = rewards + self.gamma * (1 - dones) * v_tgt
            else:
                y_q = rewards + self.gamma * (1 - dones) * (q_ns - alpha * logp_ns)

        # Critic update
        q1_s, q2_s = self.critic(states, actions)
        critic_loss = F.mse_loss(q1_s, y_q) + F.mse_loss(q2_s, y_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Value update
        if self.use_value_net:
            v_s = self.value_net(states)
            with torch.no_grad():
                mu_s, log_std_s = self.actor(states)
                std_s = log_std_s.exp()
                dist2 = torch.distributions.Normal(mu_s, std_s)
                z2 = dist2.rsample()
                a_s = torch.tanh(z2)
                logp_s = dist2.log_prob(z2).sum(-1, keepdim=True)
                logp_s -= (2 * (z2 - a_s) - (1 - a_s.pow(2))).sum(-1, keepdim=True)
                q1_s2, q2_s2 = self.critic(states, a_s)
                q_s2 = torch.min(q1_s2, q2_s2)
                a_val = q_s2 - alpha * logp_s
            value_loss = F.mse_loss(v_s, a_val)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # Actor update
        mu_s3, log_std_s3 = self.actor(states)
        std_s3 = log_std_s3.exp()
        dist3 = torch.distributions.Normal(mu_s3, std_s3)
        z3 = dist3.rsample()
        a_s3 = torch.tanh(z3)
        logp_s3 = dist3.log_prob(z3).sum(-1, keepdim=True)
        logp_s3 -= (2 * (z3 - a_s3) - (1 - a_s3.pow(2))).sum(-1, keepdim=True)
        q1_pi, q2_pi = self.critic(states, a_s3)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * logp_s3 - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logp_s3 + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update targets
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        if self.use_value_net:
            for p, tp in zip(self.value_net.parameters(), self.value_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # Debug logs
        if self.debug:
            print(f"[SACAgent] critic_loss={critic_loss.item():.4f}, actor_loss={actor_loss.item():.4f}")

    def _get_alpha(self) -> float:
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return self.alpha

    def save(self, path: str) -> None:
        ckpt = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
        if self.use_value_net:
            ckpt["value_net"] = self.value_net.state_dict()
        if self.automatic_entropy_tuning:
            ckpt["log_alpha"] = self.log_alpha.detach().cpu().numpy()
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
        if self.use_value_net:
            self.value_net.load_state_dict(ckpt["value_net"])
            self.value_target = copy.deepcopy(self.value_net)
        if self.automatic_entropy_tuning and "log_alpha" in ckpt:
            val = ckpt["log_alpha"]
            self.log_alpha = torch.tensor(val, requires_grad=True, device=self.device)
