"""
Experience Replay Buffer
========================
Off-policy replay buffer storing transition tuples ⟨s, a, r, s', done⟩.

Used by DDPG, D3PG, TD3 and MADDPG (each agent has its own buffer,
or one shared buffer depending on the setup).
"""

import numpy as np
import torch
from collections import deque
from typing import Tuple, NamedTuple


class Batch(NamedTuple):
    obs:      torch.Tensor    # (batch, obs_dim)
    action:   torch.Tensor    # (batch, action_dim)
    reward:   torch.Tensor    # (batch, 1)
    next_obs: torch.Tensor    # (batch, obs_dim)
    done:     torch.Tensor    # (batch, 1)  — float: 1.0 = terminal


class ReplayBuffer:
    """
    Fixed-size circular buffer. Samples uniformly at random.
    All tensors moved to `device` when sampled.
    """

    def __init__(self, capacity: int = 100_000, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self._obs:      deque = deque(maxlen=capacity)
        self._action:   deque = deque(maxlen=capacity)
        self._reward:   deque = deque(maxlen=capacity)
        self._next_obs: deque = deque(maxlen=capacity)
        self._done:     deque = deque(maxlen=capacity)

    def push(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        self._obs.append(obs.astype(np.float32))
        self._action.append(np.array(action, dtype=np.float32))
        self._reward.append(np.float32(reward))
        self._next_obs.append(next_obs.astype(np.float32))
        self._done.append(np.float32(done))

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, len(self), size=batch_size)

        obs      = torch.as_tensor(np.array([self._obs[i]      for i in idx])).to(self.device)
        action   = torch.as_tensor(np.array([self._action[i]   for i in idx])).to(self.device)
        reward   = torch.as_tensor(np.array([self._reward[i]   for i in idx])).unsqueeze(1).to(self.device)
        next_obs = torch.as_tensor(np.array([self._next_obs[i] for i in idx])).to(self.device)
        done     = torch.as_tensor(np.array([self._done[i]     for i in idx])).unsqueeze(1).to(self.device)

        return Batch(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)

    def __len__(self) -> int:
        return len(self._obs)

    def ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size
