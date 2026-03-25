"""
Benchmark Agents
================
Implements the two non-DDPG benchmarks from the paper (Section 4.4):

  RBMAgent    – Rule-Based Model (Section 4.4.2)
                Always maximises RES utilisation.
                Priority order: SC → LIB → VRB.
                Ignores wholesale price dynamics.

  MADQNAgent  – Multi-Agent Deep Q-Network (Section 4.4.1, Tampuu et al.)
                Each ESS agent independently learns an action-value function.
                Discrete action space (paper: 5 actions for MADQN, 9 for Rainbow).
                Uses Double DQN + experience replay.
                Benchmark-only — does not model the xMG auction.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# RBM — Rule-Based Model  (Section 4.4.2)
# -------------------------------------------------------------------------

class RBMAgent:
    """
    Rule-Based Model: charge ESS when RES > demand, discharge otherwise.
    Priority order matches paper: SC first, LIB second, VRB third.
    Returns continuous actions in [-1, 1] (converted to MW in the environment).
    """

    ESS_ORDER = [2, 0, 1]   # SC=idx2, LIB=idx0, VRB=idx1 in the env's ESS list

    def select_actions(
        self,
        demand: float,      # MW
        pv_mw: float,       # MW
        wt_mw: float,       # MW
        ess_charges: List[float],   # current [c_LIB, c_VRB, c_SC] in MWh
        ess_c_max: float = 2.0,     # max capacity per ESS
        ess_x_max: float = 1.0,     # max power per ESS
    ) -> List[float]:
        """
        Returns actions for [LIB, VRB, SC] in [-1, 1] domain.
        Positive action = charge, negative = discharge.
        """
        surplus = pv_mw + wt_mw - demand   # positive = excess RES
        actions = [0.0, 0.0, 0.0]          # default: idle

        if surplus > 0:
            # Charge ESS in priority order to absorb surplus
            remaining = surplus
            for idx in self.ESS_ORDER:
                room = ess_c_max - ess_charges[idx]
                if room > 0 and remaining > 0:
                    charge_mw = min(remaining, ess_x_max, room)
                    actions[idx] = charge_mw / ess_x_max   # normalise to [-1,1]
                    remaining -= charge_mw
        else:
            # Discharge ESS in priority order to cover deficit
            deficit = -surplus
            for idx in self.ESS_ORDER:
                avail = ess_charges[idx]
                if avail > 0 and deficit > 0:
                    discharge_mw = min(deficit, ess_x_max, avail)
                    actions[idx] = -(discharge_mw / ess_x_max)
                    deficit -= discharge_mw

        return actions


# -------------------------------------------------------------------------
# MADQN — Multi-Agent DQN  (Section 4.4.1)
# -------------------------------------------------------------------------

class _DQNNetwork(nn.Module):
    """Simple two-hidden-layer DQN (256 units, ReLU)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DQNSingleAgent:
    """
    Independent DQN for one ESS agent (Tampuu et al. approach).
    Treats all other agents as part of the environment.
    Uses Double DQN for stable learning.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 5,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 5_000,
        device: str = "cpu",
    ):
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.device     = device
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self._step      = 0

        self.online  = _DQNNetwork(obs_dim, n_actions, hidden_dim).to(device)
        self.target  = copy.deepcopy(self.online)
        self.target.eval()
        self.opt     = torch.optim.Adam(self.online.parameters(), lr=lr)

        self.buffer: deque = deque(maxlen=buffer_size)

    # ---- ε-greedy action selection -----------------------------------
    @property
    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self._step / self.eps_decay)

    def select_action(self, obs: np.ndarray) -> int:
        """Return discrete action index 0..n_actions-1."""
        self._step += 1
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return int(self.online(t).argmax(dim=1).item())

    # ---- Replay buffer -----------------------------------------------
    def push(self, obs, action: int, reward: float, next_obs, done: bool):
        self.buffer.append((
            np.array(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_obs, dtype=np.float32),
            float(done),
        ))

    # ---- Training step -----------------------------------------------
    def update(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)

        obs_t      = torch.FloatTensor(np.array(obs_b)).to(self.device)
        act_t      = torch.LongTensor(act_b).unsqueeze(1).to(self.device)
        rew_t      = torch.FloatTensor(rew_b).unsqueeze(1).to(self.device)
        next_obs_t = torch.FloatTensor(np.array(next_obs_b)).to(self.device)
        done_t     = torch.FloatTensor(done_b).unsqueeze(1).to(self.device)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.online(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs_t).gather(1, next_actions)
            target_q = rew_t + (1.0 - done_t) * self.gamma * next_q

        current_q = self.online(obs_t).gather(1, act_t)
        loss = F.mse_loss(current_q, target_q)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step()

        # Soft target update
        for p, tp in zip(self.online.parameters(), self.target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return float(loss.item())


def _discrete_action_to_continuous(idx: int, n_actions: int = 5) -> float:
    """
    Map discrete action index to continuous charge/discharge fraction in [-1, 1].
    n_actions=5 → {-1, -0.5, 0, 0.5, 1}
    n_actions=9 → evenly spaced {-1, -0.75, ..., 0.75, 1}
    """
    return -1.0 + 2.0 * idx / (n_actions - 1)


class MADQNSystem:
    """
    Multi-Agent DQN system: one independent DQN per ESS agent (3 agents).
    For case_study=1 (HESS only); does not model the MGA/xMG auction.
    Paper uses 5 discrete actions (Section 4.4.1).
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int = 3,
        n_actions: int = 5,
        device: str = "cpu",
    ):
        self.n_agents  = n_agents
        self.n_actions = n_actions
        self.agents = [
            _DQNSingleAgent(obs_dim, n_actions=n_actions, device=device)
            for _ in range(n_agents)
        ]

    def select_actions(self, obs_list: List[np.ndarray]) -> List[float]:
        """Return continuous actions in [-1, 1] for each ESS agent."""
        actions = []
        for i, agent in enumerate(self.agents):
            idx = agent.select_action(obs_list[i])
            actions.append(_discrete_action_to_continuous(idx, self.n_actions))
        return actions

    def push(
        self,
        obs_list: List[np.ndarray],
        actions_cont: List[float],
        rewards: List[float],
        next_obs_list: List[np.ndarray],
        done: bool,
    ):
        """Store transition. Converts continuous back to nearest discrete index."""
        for i, agent in enumerate(self.agents):
            # Re-derive the discrete action that was used
            cont = actions_cont[i]
            idx = int(round((cont + 1.0) / 2.0 * (self.n_actions - 1)))
            idx = max(0, min(self.n_actions - 1, idx))
            agent.push(obs_list[i], idx, rewards[i], next_obs_list[i], done)

    def update(self):
        for agent in self.agents:
            agent.update()
