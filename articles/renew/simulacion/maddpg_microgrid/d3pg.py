"""
D3PG — Distributional DDPG
============================
Extends DDPG by replacing the scalar critic with a distributional critic that
models the full return distribution Z(s, a) discretised over N_atoms atoms
in the range [V_min, V_max].

Key differences from DDPG (Section 3.2.2):
  1. Critic outputs log_softmax probabilities over N_atoms support points.
  2. Training target is the Bellman-projected target distribution.
  3. Loss is KL divergence, not MSE.

References: D3PG (Barth-Maron et al. 2018), C51 categorical algorithm (Bellemare et al. 2017).
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from .networks import Actor, DistributionalCritic
from .replay_buffer import ReplayBuffer
from .config import D3PGConfig


class D3PGAgent:
    """
    Distributional DDPG agent.
    Same interface as DDPGAgent; drop-in replacement.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: Optional[D3PGConfig] = None,
        device: str = "cpu",
    ):
        self.cfg = cfg or D3PGConfig()
        self.device = device

        # Support z_i = V_min + (i/(N_atoms−1)) * (V_max − V_min)
        self.support = torch.linspace(
            self.cfg.V_min, self.cfg.V_max, self.cfg.n_atoms
        ).to(device)                           # shape (N_atoms,)
        self.delta_z = (self.cfg.V_max - self.cfg.V_min) / (self.cfg.n_atoms - 1)

        # ---- Actor: online + target ----
        self.actor        = Actor(obs_dim, action_dim, self.cfg.hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)

        # ---- Distributional critic: online + target ----
        self.critic        = DistributionalCritic(obs_dim, action_dim, self.cfg.n_atoms, self.cfg.hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        # ---- Replay buffer ----
        self.buffer = ReplayBuffer(self.cfg.buffer_size, device)

    # ------------------------------------------------------------------
    # Action selection (identical to DDPG)
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
        B = cfg.batch_size

        # ---- Build projected target distribution (categorical algorithm) ----
        with torch.no_grad():
            # Next actions from target actor
            self.actor_target.eval()
            next_a = self.actor_target(batch.next_obs)                   # (B, A)

            # Target critic probabilities for next state
            next_log_probs = self.critic_target(batch.next_obs, next_a)  # (B, N)
            next_probs     = next_log_probs.exp()                        # (B, N)

            # Compute projected atoms: ẑ_j = r + γ * z_j  (clipped to [V_min, V_max])
            # batch.reward: (B,1), self.support: (N,)
            z_j = batch.reward + cfg.gamma * self.support.unsqueeze(0) * (1.0 - batch.done)  # (B, N)
            z_j = z_j.clamp(cfg.V_min, cfg.V_max)                                             # (B, N)

            # Project onto fixed support: distribute probability mass between
            # the two neighbouring atoms using linear interpolation
            b_j = (z_j - cfg.V_min) / self.delta_z                     # (B, N) float index
            lower = b_j.floor().long().clamp(0, cfg.n_atoms - 1)        # (B, N)
            upper = b_j.ceil().long().clamp(0, cfg.n_atoms - 1)         # (B, N)

            # Handle exact atom hits
            same = (upper == lower)
            upper[same] = (upper[same] + 1).clamp(max=cfg.n_atoms - 1)
            lower[same[upper == lower]] = (lower[same[upper == lower]] - 1).clamp(min=0)  # rare edge

            # Build projected distribution m shape (B, N_atoms)
            m = torch.zeros(B, cfg.n_atoms, device=self.device)
            offset = torch.arange(0, B * cfg.n_atoms, cfg.n_atoms, device=self.device)\
                         .unsqueeze(1).expand_as(b_j)

            m.view(-1).scatter_add_(
                0,
                (lower + offset).view(-1),
                (next_probs * (upper.float() - b_j)).view(-1),
            )
            m.view(-1).scatter_add_(
                0,
                (upper + offset).view(-1),
                (next_probs * (b_j - lower.float())).view(-1),
            )
            # m is now the target distribution Φ·Ẑ_θ̂(s', a')

        # ---- Critic loss: KL(target ‖ predicted)  ----
        log_probs = self.critic(batch.obs, batch.action)    # (B, N) log-softmax
        # KL(m ‖ p) = Σ m * log(m / p) = Σ m * (log(m) − log(p))
        # We minimise −Σ m * log(p)  (cross-entropy, equivalent up to constant)
        critic_loss = -(m * log_probs).sum(dim=-1).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- Actor loss: −E[Q(s, μ(s))]  (use expected Q from distribution) ----
        self.actor.sample_noise()
        self.actor.train()
        a_pred = self.actor(batch.obs)
        q_values = self.critic.q_value(batch.obs, a_pred, self.support)  # (B,)
        actor_loss = -q_values.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- Soft target updates ----
        DDPGSoftUpdate = _soft_update
        DDPGSoftUpdate(self.critic, self.critic_target, cfg.tau)
        DDPGSoftUpdate(self.actor,  self.actor_target,  cfg.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
        }

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


def _soft_update(online: torch.nn.Module, target: torch.nn.Module, tau: float):
    for p, p_tgt in zip(online.parameters(), target.parameters()):
        p_tgt.data.copy_(tau * p.data + (1.0 - tau) * p_tgt.data)
