"""
Neural Network Architectures
=============================
Implements the networks used by DDPG, D3PG, TD3 and MADDPG:

  - NoisyLinear   — NoisyNet layer (replaces ε-greedy exploration for actor)
  - Actor         — deterministic policy μ_φ(s) → action ∈ (−1, 1)
  - Critic        — standard Q-value network  Q_θ(s, a) → scalar
  - DistributionalCritic — D3PG: outputs atom probabilities  Z_θ(s, a)
  - MADDPGCritic  — multi-agent critic: Q_θ(s, a_all) → scalar
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# NoisyNet linear layer  (used in Actor to replace ε-greedy, per paper)
# ---------------------------------------------------------------------------

class NoisyLinear(nn.Module):
    """
    Factorised NoisyNet linear layer (Fortunato et al. 2017).
    The actor uses this instead of adding an external noise process.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init

        # Learnable parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Persistent noise (re-sampled each forward pass)
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps",   torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ---------------------------------------------------------------------------
# Helper: build a standard MLP body
# ---------------------------------------------------------------------------

def _mlp_body(input_dim: int, hidden_dim: int) -> nn.Sequential:
    """Two hidden layers with ReLU (256 units each, per Table 2)."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
    )


# ---------------------------------------------------------------------------
# Actor  μ_φ(s) → a ∈ (−1, 1)^n_actions
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Deterministic actor network.
    The last layer is a NoisyLinear layer so that exploration is
    handled by the network itself (no external noise process needed).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.body = _mlp_body(obs_dim, hidden_dim)
        self.head = NoisyLinear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.body(obs)))

    def sample_noise(self):
        self.head.sample_noise()


# ---------------------------------------------------------------------------
# Critic  Q_θ(s, a) → scalar
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    Standard Q-value critic.
    State and action are concatenated and passed through two hidden layers.
    Used by DDPG, TD3, and MADDPG.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


# ---------------------------------------------------------------------------
# MADDPG Critic  Q_θ(s_all, a_all) → scalar
# ---------------------------------------------------------------------------

class MADDPGCritic(nn.Module):
    """
    Centralised critic for MADDPG.
    Receives concatenated observations AND actions of ALL agents.
    """

    def __init__(self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_all: torch.Tensor, actions_all: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs_all, actions_all], dim=-1))


# ---------------------------------------------------------------------------
# D3PG Distributional Critic  Z_θ(s, a) → probability distribution over atoms
# ---------------------------------------------------------------------------

class DistributionalCritic(nn.Module):
    """
    Distributional critic for D3PG (C51-style).
    Outputs a softmax probability distribution over N_atoms support points.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_atoms: int = 51,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_atoms),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns log-softmax probabilities over atoms. Shape: (batch, n_atoms)."""
        logits = self.net(torch.cat([obs, action], dim=-1))
        return F.log_softmax(logits, dim=-1)

    def probs(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities (not log). Shape: (batch, n_atoms)."""
        return self.forward(obs, action).exp()

    def q_value(
        self, obs: torch.Tensor, action: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        """Scalar Q = sum(p_i * z_i). Shape: (batch,)."""
        probs = self.probs(obs, action)          # (batch, n_atoms)
        return (probs * support.unsqueeze(0)).sum(dim=-1)
