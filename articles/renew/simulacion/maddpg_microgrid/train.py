"""
Training Script
===============
Runs the full training loop described in Section 4.3 of the paper.

Supports all algorithm modes:
  - DDPG  / D3PG  / TD3   (single-agent)
  - MADDPG / MAD3PG / MATD3 (multi-agent via MADDPGSystem)

Usage
-----
    python -m src.maddpg_microgrid.train --algo maddpg --case 2 --episodes 200

or from Python:
    from src.maddpg_microgrid.train import run
    run(algo="matd3", case_study=2, total_episodes=200)
"""

import argparse
import time
from typing import Optional, Dict, Any
import numpy as np
import torch

from .environment import HESSMicrogridEnv, SyntheticDataGenerator
from .config import DDPGConfig, D3PGConfig, TD3Config, SimConfig
from .ddpg import DDPGAgent
from .d3pg import D3PGAgent
from .td3 import TD3Agent
from .maddpg import MADDPGSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_sgc_agent(algo: str, obs_dim: int, action_dim: int):
    """Build a single-agent controller."""
    algo = algo.lower()
    if algo == "ddpg":
        return DDPGAgent(obs_dim, action_dim, DDPGConfig(), device=DEVICE)
    if algo == "d3pg":
        return D3PGAgent(obs_dim, action_dim, D3PGConfig(), device=DEVICE)
    if algo == "td3":
        return TD3Agent(obs_dim, action_dim, TD3Config(), device=DEVICE)
    raise ValueError(f"Unknown single-agent algorithm: {algo!r}")


def _make_mas_system(algo: str, env: HESSMicrogridEnv):
    """Build a MADDPGSystem for the given case study."""
    algo = algo.lower()
    obs_dim_ess = env.obs_size_ess()
    obs_dim_xmg = env.obs_size_xmg()

    if env.case_study == 1:
        obs_dims    = [obs_dim_ess] * 3
        action_dims = [1, 1, 1]             # one action per ESS (charge/discharge)
    else:
        obs_dims    = [obs_dim_ess] * 3 + [obs_dim_ess] + [obs_dim_xmg] * 5
        action_dims = [1, 1, 1, 2, 2, 2, 2, 2, 2]   # ESS×3(1), MGA(2), xMG×5(2) → 15 total

    base = algo.replace("ma", "").replace("matd3", "td3").replace("mad3pg", "d3pg").replace("maddpg", "ddpg")
    if "td3" in algo:
        cfg = TD3Config()
    elif "d3pg" in algo:
        cfg = D3PGConfig()
    else:
        cfg = DDPGConfig()

    return MADDPGSystem(obs_dims, action_dims, algorithm=base, cfg=cfg, device=DEVICE)


# ---------------------------------------------------------------------------
# Single-agent training loop
# ---------------------------------------------------------------------------

def _train_sgc(
    agent,
    env: HESSMicrogridEnv,
    sim_cfg: SimConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a single-agent controller (DDPG / D3PG / TD3)."""
    global_step   = 0
    episode_savings = []

    for episode in range(sim_cfg.total_episodes):
        obs_list = env.reset()
        obs = obs_list[0]   # single agent observes ESS state
        episode_reward = 0.0
        total_r_sum = 0.0

        for _ in range(sim_cfg.steps_per_episode):
            global_step += 1

            # Action selection
            if global_step < sim_cfg.random_action_steps:
                action = np.random.uniform(-1, 1, size=(env.n_agents if env.case_study == 1 else 3,))
                if env.case_study == 2:
                    action = np.random.uniform(-1, 1, size=(3 + 2,))  # ESS + MGA
            else:
                action = agent.select_action(obs)

            # Build full action list for environment
            action_list = list(action) if env.case_study == 1 else list(action[:5]) + \
                          list(np.random.uniform(-1, 1, 10))  # random xMG actions in SGC

            next_obs_list, rewards, done, info = env.step(action_list)
            next_obs = next_obs_list[0]
            reward   = rewards[0]

            agent.push(obs, action, reward, next_obs, float(done))

            if global_step >= sim_cfg.learning_start_steps:
                agent.update()

            obs = next_obs
            episode_reward += reward
            total_r_sum    += info["R_sum"]

            if done:
                break

        # Track savings: only during evaluation phase
        savings_gbp = total_r_sum / 0.01 / env.n_agents  # reverse normalisation
        episode_savings.append(savings_gbp)

        if verbose and episode % 10 == 0:
            mean_sav = np.mean(episode_savings[-10:]) if len(episode_savings) >= 10 else episode_savings[-1]
            print(f"[Episode {episode:4d}]  reward={episode_reward:+.3f}  "
                  f"savings=£{savings_gbp:,.0f}/week  "
                  f"mean10=£{mean_sav:,.0f}/week")

    return {"episode_savings": episode_savings}


# ---------------------------------------------------------------------------
# Multi-agent training loop
# ---------------------------------------------------------------------------

def _train_mas(
    system: MADDPGSystem,
    env: HESSMicrogridEnv,
    sim_cfg: SimConfig,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train the MADDPG multi-agent system."""
    global_step   = 0
    episode_savings = []
    eval_savings    = []

    for episode in range(sim_cfg.total_episodes):
        obs_list = env.reset()
        episode_reward_sum = 0.0
        total_r_sum = 0.0

        for _ in range(sim_cfg.steps_per_episode):
            global_step += 1

            if global_step < sim_cfg.random_action_steps:
                actions = [np.random.uniform(-1, 1, d) for d in system.action_dims]
            else:
                actions = system.select_actions(obs_list)

            # Flatten actions to list of floats for environment
            flat_actions = [float(a) if np.ndim(a) == 0 else a.tolist() for a in actions]
            flat_actions_env = []
            for a in flat_actions:
                if isinstance(a, list):
                    flat_actions_env.extend(a)
                else:
                    flat_actions_env.append(a)

            next_obs_list, rewards, done, info = env.step(flat_actions_env)

            system.push(obs_list, actions, rewards, next_obs_list, done)

            if global_step >= sim_cfg.learning_start_steps:
                system.update()

            obs_list = next_obs_list
            episode_reward_sum += sum(rewards)
            total_r_sum        += info["R_sum"]

            if done:
                break

        savings_gbp = total_r_sum / 0.01 / env.n_agents
        episode_savings.append(savings_gbp)

        if episode >= sim_cfg.eval_start_episode:
            eval_savings.append(savings_gbp)

        if verbose and episode % 10 == 0:
            mean_sav = np.mean(episode_savings[-10:]) if len(episode_savings) >= 10 else episode_savings[-1]
            print(f"[Episode {episode:4d}]  total_reward={episode_reward_sum:+.3f}  "
                  f"savings=£{savings_gbp:,.0f}/week  "
                  f"mean10=£{mean_sav:,.0f}/week")

    total_eval_savings = sum(eval_savings) if eval_savings else 0.0
    print(f"\n✓ Evaluation savings (episodes {sim_cfg.eval_start_episode}–{sim_cfg.total_episodes}): "
          f"£{total_eval_savings:,.0f}")

    return {
        "episode_savings": episode_savings,
        "eval_savings": eval_savings,
        "total_eval_savings_gbp": total_eval_savings,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    algo: str = "maddpg",
    case_study: int = 2,
    total_episodes: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run training for the given algorithm and case study.

    Parameters
    ----------
    algo         : one of ddpg, d3pg, td3, maddpg, mad3pg, matd3
    case_study   : 1 = ESS only,  2 = ESS + xMG trading
    total_episodes : number of training episodes (default 200 = paper)
    seed         : random seed
    verbose      : print episode results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    sim_cfg = SimConfig(total_episodes=total_episodes)
    data_gen = SyntheticDataGenerator(seed=seed)
    env = HESSMicrogridEnv(case_study=case_study, sim_cfg=sim_cfg, data_generator=data_gen, seed=seed)

    is_multi = algo.startswith("ma")

    print(f"{'='*60}")
    print(f"Algorithm  : {algo.upper()}")
    print(f"Case study : {case_study}")
    print(f"Episodes   : {total_episodes}  (eval from {sim_cfg.eval_start_episode})")
    print(f"Agents     : {env.n_agents}")
    print(f"Device     : {DEVICE}")
    print(f"{'='*60}\n")

    t0 = time.time()

    if is_multi:
        system = _make_mas_system(algo, env)
        results = _train_mas(system, env, sim_cfg, verbose)
    else:
        # Single-agent: one agent controls all its ESS outputs
        obs_dim    = env.obs_size_ess()
        action_dim = 3 if case_study == 1 else 5   # 3 ESS or 3 ESS + 2 MGA
        agent = _make_sgc_agent(algo, obs_dim, action_dim)
        results = _train_sgc(agent, env, sim_cfg, verbose)

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MADDPG microgrid agent")
    parser.add_argument("--algo",     type=str, default="maddpg",
                        choices=["ddpg","d3pg","td3","maddpg","mad3pg","matd3"])
    parser.add_argument("--case",     type=int, default=2, choices=[1, 2])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    run(
        algo=args.algo,
        case_study=args.case,
        total_episodes=args.episodes,
        seed=args.seed,
        verbose=not args.quiet,
    )
