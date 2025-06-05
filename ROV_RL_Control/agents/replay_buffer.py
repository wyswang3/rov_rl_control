# 文件：rov_rl_control/agents/replay_buffer.py

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    通用经验回放缓冲区，存储 (state, action, reward, next_state, done)。
    支持环形存储，当容量满时覆盖最早的数据；支持随机小批量采样。

    Attributes:
        max_size (int): 缓冲区最大容量
        state_dim (int): 状态维度
        action_dim (int): 动作维度
        ptr (int): 当前写入位置指针
        size (int): 当前缓冲区中存储的样本数量（<= max_size）
        states (np.ndarray): 存储状态，形状 (max_size, state_dim)
        actions (np.ndarray): 存储动作，形状 (max_size, action_dim)
        rewards (np.ndarray): 存储即时奖励，形状 (max_size, 1)
        next_states (np.ndarray): 存储下一状态，形状 (max_size, state_dim)
        dones (np.ndarray): 存储 done 标志，形状 (max_size, 1)，使用 0/1 表示
    """

    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int) -> None:
        """
        Args:
            max_size (int): 缓冲区最大容量
            state_dim (int): 状态维度
            action_dim (int): 动作维度
        """
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ptr = 0
        self.size = 0

        # 预分配内存
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool) -> None:
        """
        向缓冲区添加一个样本 (s, a, r, s', done)。

        Args:
            state (np.ndarray): 当前状态，shape = (state_dim,)
            action (np.ndarray): 当前动作，shape = (action_dim,)
            reward (float): 即时奖励
            next_state (np.ndarray): 下一状态，shape = (state_dim,)
            done (bool): 是否结束标志
        """
        # 存储数据
        self.states[self.ptr] = state.copy()
        self.actions[self.ptr] = action.copy()
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state.copy()
        self.dones[self.ptr] = float(done)

        # 更新指针与当前容量
        self.ptr = (self.ptr + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def sample(self,
               batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从缓冲区随机采样一个 minibatch。

        Args:
            batch_size (int): 样本数量

        Returns:
            Tuple of:
                states (np.ndarray): shape = (batch_size, state_dim)
                actions (np.ndarray): shape = (batch_size, action_dim)
                rewards (np.ndarray): shape = (batch_size, 1)
                next_states (np.ndarray): shape = (batch_size, state_dim)
                dones (np.ndarray): shape = (batch_size, 1)
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                self.next_states[idxs],
                self.dones[idxs])

    def __len__(self) -> int:
        """返回缓冲区当前存储的样本数量"""
        return self.size
