"""
DDPG — Deep Deterministic Policy Gradient
==========================================
Implements DDPG (Lillicrap et al. 2016) as described in Section 3.2.1
of the paper, including:
  - NoisyNet actor (replaces OU/Gaussian noise process)
  - Experience replay buffer
  - Soft target network updates  (Eq. soft-update)
  - Critic loss (Eq. L_i)  and  actor gradient (Eq. ∇_φ J_i)
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer
from .config import DDPGConfig


class DDPGAgent:
    """
    Single DDPG agent.

    Can be used as a standalone single-agent controller (SGC)
    or as one agent inside the MADDPG system (with a different critic).

    Parameters
    ----------
    obs_dim    : dimension of LOCAL observation
    action_dim : number of continuous actions produced
    cfg        : DDPGConfig hyperparameters
    device     : torch device string
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: Optional[DDPGConfig] = None,
        device: str = "cpu",
    ):
        self.cfg = cfg or DDPGConfig()
        self.device = device
        self.action_dim = action_dim

        # ---- Actor: online + target ----
        self.actor        = Actor(obs_dim, action_dim, self.cfg.hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)

        # ---- Critic: online + target ----
        self.critic        = Critic(obs_dim, action_dim, self.cfg.hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        # ---- Replay buffer ----
        self.buffer = ReplayBuffer(self.cfg.buffer_size, device)

        self._step_count = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Returns action ∈ (−1, 1)^action_dim.
        NoisyNet provides exploration when training (deterministic=False).
        """
        if not deterministic:
            self.actor.sample_noise()

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(obs_t).squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Store experience
    # ------------------------------------------------------------------

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def update(self) -> Optional[dict]:
        """
        One gradient step. Returns dict with loss values, or None if buffer
        is not yet large enough.
        """
        cfg = self.cfg
        if not self.buffer.ready(cfg.batch_size):
            return None

        batch = self.buffer.sample(cfg.batch_size)
        self._step_count += 1

        # ---- Critic update ----
        with torch.no_grad():
            self.actor_target.eval()
            next_actions = self.actor_target(batch.next_obs)          # μ̂_φ̂(s')
            q_next = self.critic_target(batch.next_obs, next_actions)  # Q̂_θ̂(s', μ̂(s'))
            # Target (Eq. y_i = r_i + γ·Q̂·(1 − done))
            y = batch.reward + cfg.gamma * q_next * (1.0 - batch.done)

        q_pred = self.critic(batch.obs, batch.action)
        critic_loss = F.mse_loss(q_pred, y)           # L_i(θ) = E[(y_i − Q_θ)²]

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- Actor update ----
        # Resample noise for the online actor
        self.actor.sample_noise()
        self.actor.train()

        actions_pred = self.actor(batch.obs)
        # Actor loss = −E[Q_θ(s, μ_φ(s))]   (policy gradient via gradient ascent)
        actor_loss = -self.critic(batch.obs, actions_pred).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- Soft target updates: θ̂ ← τ·θ + (1−τ)·θ̂ ----
        self._soft_update(self.critic, self.critic_target, cfg.tau)
        self._soft_update(self.actor,  self.actor_target,  cfg.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _soft_update(online: torch.nn.Module, target: torch.nn.Module, tau: float):
        for p, p_tgt in zip(online.parameters(), target.parameters()):
            p_tgt.data.copy_(tau * p.data + (1.0 - tau) * p_tgt.data)

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
