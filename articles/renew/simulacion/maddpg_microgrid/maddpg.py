"""
MADDPG — Multi-Agent Deep Deterministic Policy Gradient
=========================================================
Implements the centralised-learning / decentralised-execution paradigm
described in Section 3.3 and built on top of DDPG/D3PG/TD3.

Key design decisions from the paper:
  - Each agent has its OWN actor that observes only its LOCAL state.
  - Each agent has its OWN centralised critic that receives the concatenated
    observations AND actions of ALL agents during training.
  - Each agent has its OWN reward function (marginal contribution).
  - The base algorithm (actor updates) can be DDPG, D3PG, or TD3 per-agent.

Usage
-----
    sys = MADDPGSystem(obs_dims, action_dims, algorithm="td3")
    sys.push(obs_list, action_list, reward_list, next_obs_list, done)
    losses = sys.update()
    actions = sys.select_actions(obs_list)
"""

import copy
from typing import List, Optional, Dict, Union
import numpy as np
import torch
import torch.nn.functional as F

from .networks import Actor, MADDPGCritic
from .replay_buffer import ReplayBuffer, Batch
from .config import DDPGConfig, TD3Config, D3PGConfig


# ---------------------------------------------------------------------------
# Single MADDPG agent wrapper
# ---------------------------------------------------------------------------

class _MADDPGAgent:
    """
    One agent inside the MADDPG system.
    - Local actor:          obs_i  → action_i
    - Centralised critic:   (obs_all, actions_all) → Q_i
    """

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        action_dim: int,
        total_obs_dim: int,
        total_action_dim: int,
        cfg: DDPGConfig,
        device: str,
        actor_delay: int = 1,    # 1 = DDPG/D3PG, 2 = TD3
    ):
        self.agent_id   = agent_id
        self.cfg        = cfg
        self.device     = device
        self.actor_delay = actor_delay
        self._update_count = 0

        # Local actor
        self.actor        = Actor(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        # Centralised critic (uses global state + global actions)
        self.critic        = MADDPGCritic(total_obs_dim, total_action_dim, cfg.hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def update_targets(self):
        _soft_update(self.actor,  self.actor_target,  self.cfg.tau)
        _soft_update(self.critic, self.critic_target, self.cfg.tau)


# ---------------------------------------------------------------------------
# MADDPG system
# ---------------------------------------------------------------------------

class MADDPGSystem:
    """
    Full MADDPG multi-agent system.

    Parameters
    ----------
    obs_dims        : list of observation dimensions, one per agent
    action_dims     : list of action dimensions, one per agent
    algorithm       : "ddpg" | "d3pg" (not yet implemented, uses DDPG actor) | "td3"
    cfg             : shared DDPGConfig / TD3Config hyperparameters
    device          : torch device string
    """

    def __init__(
        self,
        obs_dims: List[int],
        action_dims: List[int],
        algorithm: str = "ddpg",
        cfg: Optional[DDPGConfig] = None,
        device: str = "cpu",
    ):
        self.n_agents    = len(obs_dims)
        self.obs_dims    = obs_dims
        self.action_dims = action_dims
        self.device      = device
        self.algorithm   = algorithm.lower()

        if cfg is None:
            cfg = TD3Config() if self.algorithm == "td3" else DDPGConfig()
        self.cfg = cfg

        actor_delay = getattr(cfg, "actor_delay", 1)

        # Centralised dimensions
        self.total_obs_dim    = sum(obs_dims)
        self.total_action_dim = sum(action_dims)

        # One agent per agent index
        self.agents: List[_MADDPGAgent] = [
            _MADDPGAgent(
                agent_id=i,
                obs_dim=obs_dims[i],
                action_dim=action_dims[i],
                total_obs_dim=self.total_obs_dim,
                total_action_dim=self.total_action_dim,
                cfg=cfg,
                device=device,
                actor_delay=actor_delay,
            )
            for i in range(self.n_agents)
        ]

        # Shared replay buffer storing centralised transitions
        # Each entry: (obs_all, actions_all, rewards_all, next_obs_all, done)
        self._shared_buffer = _CentralisedBuffer(cfg.buffer_size)
        self._update_count = 0

        # For TD3: track per-agent policy noise
        self._policy_noise = getattr(cfg, "policy_noise", 0.2)
        self._noise_clip   = getattr(cfg, "noise_clip", 0.5)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_actions(
        self, obs_list: List[np.ndarray], deterministic: bool = False
    ) -> List[np.ndarray]:
        """
        Each agent observes only its own local observation.
        Returns list of actions, one per agent.
        """
        actions = []
        for agent, obs in zip(self.agents, obs_list):
            if not deterministic:
                agent.actor.sample_noise()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = agent.actor(obs_t).squeeze(0).cpu().numpy()
            actions.append(a)
        return actions

    # ------------------------------------------------------------------
    # Store experience
    # ------------------------------------------------------------------

    def push(
        self,
        obs_list:      List[np.ndarray],
        actions_list:  List[np.ndarray],
        rewards_list:  List[float],
        next_obs_list: List[np.ndarray],
        done: bool,
    ):
        self._shared_buffer.push(obs_list, actions_list, rewards_list, next_obs_list, done)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self) -> Optional[Dict[str, float]]:
        cfg = self.cfg
        if len(self._shared_buffer) < cfg.batch_size:
            return None

        self._update_count += 1
        batch = self._shared_buffer.sample(cfg.batch_size, self.device)
        # batch keys: obs_all (B, total_obs), actions_all (B, total_action),
        #             rewards (B, n_agents), next_obs_all, done (B,1)

        losses = {}

        # ---- Compute target actions from all target actors ----
        with torch.no_grad():
            target_actions_all = []
            offset = 0
            for agent in self.agents:
                obs_i = batch["next_obs_all"][:, offset: offset + self.obs_dims[agent.agent_id]]
                offset += self.obs_dims[agent.agent_id]

                agent.actor_target.eval()
                a_next = agent.actor_target(obs_i)

                # TD3: add clipped noise to target actions
                if self.algorithm == "td3":
                    noise = (torch.randn_like(a_next) * self._policy_noise).clamp(
                        -self._noise_clip, self._noise_clip
                    )
                    a_next = (a_next + noise).clamp(-1.0, 1.0)

                target_actions_all.append(a_next)
            target_actions_cat = torch.cat(target_actions_all, dim=-1)   # (B, total_action)

        # ---- Per-agent critic + actor update ----
        for agent in self.agents:
            i = agent.agent_id

            # ---- Critic update ----
            with torch.no_grad():
                q_next = agent.critic_target(batch["next_obs_all"], target_actions_cat)
                y = batch["rewards"][:, i:i+1] + cfg.gamma * q_next * (1.0 - batch["done"])

            q_pred = agent.critic(batch["obs_all"], batch["actions_all"])
            critic_loss = F.mse_loss(q_pred, y)

            agent.critic_opt.zero_grad()
            critic_loss.backward()
            agent.critic_opt.step()

            losses[f"critic_loss_{i}"] = critic_loss.item()

            # ---- Actor update (delayed for TD3) ----
            agent._update_count += 1
            if agent._update_count % agent.actor_delay == 0:
                # Recompute actions for this agent with gradient, others detached
                online_actions = []
                obs_offset = 0
                for j, ag_j in enumerate(self.agents):
                    obs_j = batch["obs_all"][:, obs_offset: obs_offset + self.obs_dims[j]]
                    obs_offset += self.obs_dims[j]
                    if j == i:
                        ag_j.actor.sample_noise()
                        ag_j.actor.train()
                        a_j = ag_j.actor(obs_j)          # differentiable
                    else:
                        with torch.no_grad():
                            a_j = ag_j.actor(obs_j).detach()
                    online_actions.append(a_j)

                actions_cat = torch.cat(online_actions, dim=-1)
                actor_loss = -agent.critic(batch["obs_all"], actions_cat).mean()

                agent.actor_opt.zero_grad()
                actor_loss.backward()
                agent.actor_opt.step()

                losses[f"actor_loss_{i}"] = actor_loss.item()

                # Soft target updates
                agent.update_targets()

        return losses

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path_prefix: str):
        """Save each agent to `{path_prefix}_agent_{i}.pt`."""
        for agent in self.agents:
            torch.save({
                "actor":  agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
            }, f"{path_prefix}_agent_{agent.agent_id}.pt")

    def load(self, path_prefix: str):
        for agent in self.agents:
            ckpt = torch.load(
                f"{path_prefix}_agent_{agent.agent_id}.pt",
                map_location=self.device,
            )
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
            agent.actor_target  = copy.deepcopy(agent.actor)
            agent.critic_target = copy.deepcopy(agent.critic)


# ---------------------------------------------------------------------------
# Internal centralised replay buffer
# ---------------------------------------------------------------------------

class _CentralisedBuffer:
    """Stores full joint transitions for MADDPG centralised training."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf = []
        self._pos = 0

    def push(self, obs_list, actions_list, rewards_list, next_obs_list, done):
        entry = {
            "obs":      [o.copy() for o in obs_list],
            "actions":  [np.atleast_1d(a).copy() for a in actions_list],
            "rewards":  list(rewards_list),
            "next_obs": [o.copy() for o in next_obs_list],
            "done":     float(done),
        }
        if len(self._buf) < self.capacity:
            self._buf.append(entry)
        else:
            self._buf[self._pos] = entry
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, len(self._buf), size=batch_size)
        samples = [self._buf[i] for i in indices]

        obs_all      = torch.tensor(np.array([np.concatenate(s["obs"])      for s in samples]), dtype=torch.float32).to(device)
        actions_all  = torch.tensor(np.array([np.concatenate(s["actions"])  for s in samples]), dtype=torch.float32).to(device)
        next_obs_all = torch.tensor(np.array([np.concatenate(s["next_obs"]) for s in samples]), dtype=torch.float32).to(device)
        rewards      = torch.tensor(np.array([s["rewards"]                  for s in samples]), dtype=torch.float32).to(device)
        done         = torch.tensor(np.array([[s["done"]]                   for s in samples]), dtype=torch.float32).to(device)

        return {
            "obs_all":     obs_all,
            "actions_all": actions_all,
            "next_obs_all":next_obs_all,
            "rewards":     rewards,
            "done":        done,
        }

    def __len__(self):
        return len(self._buf)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _soft_update(online: torch.nn.Module, target: torch.nn.Module, tau: float):
    for p, p_tgt in zip(online.parameters(), target.parameters()):
        p_tgt.data.copy_(tau * p.data + (1.0 - tau) * p_tgt.data)
