"""
TD3 — Twin Delayed DDPG
========================
Implements TD3 (Fujimoto et al. 2018) as described in Section 3.2.3.

Three key improvements over DDPG:
  1. Twin critics — target y_i = r + γ · min(Q̂₁, Q̂₂)   (Eq. y_i TD3)
  2. Both critics trained to the same target                (Eq. L₁, L₂)
  3. Delayed actor updates — actor updated every `actor_delay` critic steps
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer
from .config import TD3Config


class TD3Agent:
    """
    Twin Delayed DDPG agent. Same interface as DDPGAgent.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: Optional[TD3Config] = None,
        device: str = "cpu",
    ):
        self.cfg = cfg or TD3Config()
        self.device = device
        self.action_dim = action_dim

        # ---- Actor: online + target ----
        self.actor        = Actor(obs_dim, action_dim, self.cfg.hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)

        # ---- Two critics + two target critics ----
        self.critic1        = Critic(obs_dim, action_dim, self.cfg.hidden_dim).to(device)
        self.critic2        = Critic(obs_dim, action_dim, self.cfg.hidden_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic1_target.eval()
        self.critic2_target.eval()
        # Both critics share one optimiser (common in TD3 implementations)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.cfg.critic_lr,
        )

        self.buffer = ReplayBuffer(self.cfg.buffer_size, device)
        self._update_count = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic:
            self.actor.sample_noise()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(obs_t).squeeze(0).cpu().numpy()

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self) -> Optional[dict]:
        cfg = self.cfg
        if not self.buffer.ready(cfg.batch_size):
            return None

        batch = self.buffer.sample(cfg.batch_size)
        self._update_count += 1

        # ---- Compute target y_i (Eq. y_i TD3) ----
        with torch.no_grad():
            self.actor_target.eval()
            next_a = self.actor_target(batch.next_obs)

            # Target-policy smoothing: add clipped noise to target actions
            noise = (torch.randn_like(next_a) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            next_a = (next_a + noise).clamp(-1.0, 1.0)

            # Take minimum of two target critics  ← key to avoiding overestimation
            q1_next = self.critic1_target(batch.next_obs, next_a)
            q2_next = self.critic2_target(batch.next_obs, next_a)
            q_next  = torch.min(q1_next, q2_next)
            y = batch.reward + cfg.gamma * q_next * (1.0 - batch.done)

        # ---- Update both critics toward the same target (Eq. L₁, L₂) ----
        q1 = self.critic1(batch.obs, batch.action)
        q2 = self.critic2(batch.obs, batch.action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- Delayed actor update ----
        actor_loss_val = None
        if self._update_count % cfg.actor_delay == 0:
            self.actor.sample_noise()
            self.actor.train()
            a_pred = self.actor(batch.obs)
            # Policy gradient uses only critic 1  (paper Section 3.2.3)
            actor_loss = -self.critic1(batch.obs, a_pred).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_val = actor_loss.item()

            # Soft update target networks (only when actor updates)
            _soft_update(self.critic1, self.critic1_target, cfg.tau)
            _soft_update(self.critic2, self.critic2_target, cfg.tau)
            _soft_update(self.actor,   self.actor_target,   cfg.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss_val,
        }

    def save(self, path: str):
        torch.save({
            "actor":   self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)


def _soft_update(online: torch.nn.Module, target: torch.nn.Module, tau: float):
    for p, p_tgt in zip(online.parameters(), target.parameters()):
        p_tgt.data.copy_(tau * p.data + (1.0 - tau) * p_tgt.data)
