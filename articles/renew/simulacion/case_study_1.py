"""
Case Study 1 — HESS-Only ESS Control
======================================
Replicates Section 5.1 of the paper.

Scenario:
  - 3 ESS agents: LIB (idx 0), VRB (idx 1), SC (idx 2)
  - No MGA / xMG (case_study=1 in the environment)
  - 200 episodes × 168 steps = 33,600 total steps
  - First 100 episodes: training / hyperparameter phase
  - Second 100 episodes: evaluation phase
  - Reward: marginal contribution (multi-agent) or global savings baseline (single-agent)

Algorithms compared (matching Table 3 / Table 4):
  Single-agent (SGC): DDPG, D3PG, TD3
  Multi-agent  (MAS): MADDPG, MAD3PG, MATD3
  Benchmarks:         MADQN, RBM

Usage
-----
    # Run one algorithm
    from articles.renew.simulacion.case_study_1 import run_case_study_1
    results = run_case_study_1("maddpg", total_episodes=200)

    # CLI
    python case_study_1.py --algo maddpg --episodes 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# -- path bootstrap (must be first import from this package) ---------------
import sim_config  # noqa: F401  — sets sys.path to repo root

from maddpg_microgrid.environment import HESSMicrogridEnv, SyntheticDataGenerator
from maddpg_microgrid.config import DDPGConfig, D3PGConfig, TD3Config, SimConfig
from maddpg_microgrid.ddpg import DDPGAgent
from maddpg_microgrid.d3pg import D3PGAgent
from maddpg_microgrid.td3 import TD3Agent
from maddpg_microgrid.maddpg import MADDPGSystem
from benchmarks import RBMAgent, MADQNSystem

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

def _env(seed: int, total_episodes: int) -> HESSMicrogridEnv:
    eval_start   = max(1, total_episodes // 2)
    total_steps  = total_episodes * 168  # steps_per_episode = 168 h
    rand_steps   = min(1000, max(64, total_steps // 10))
    learn_steps  = min(500,  max(32, rand_steps // 2))
    sim_cfg = SimConfig(
        total_episodes=total_episodes,
        eval_start_episode=eval_start,
        random_action_steps=rand_steps,
        learning_start_steps=learn_steps,
    )
    data_gen = SyntheticDataGenerator(seed=seed)
    return HESSMicrogridEnv(case_study=1, sim_cfg=sim_cfg, data_generator=data_gen, seed=seed)


def _make_sgc(algo: str, obs_dim: int, n_actions: int = 3):
    """Single-agent controller for case study 1 (outputs 3 actions: LIB, VRB, SC)."""
    if algo == "ddpg":
        return DDPGAgent(obs_dim, n_actions, DDPGConfig(), device=DEVICE)
    if algo == "d3pg":
        return D3PGAgent(obs_dim, n_actions, D3PGConfig(), device=DEVICE)
    if algo == "td3":
        return TD3Agent(obs_dim, n_actions, TD3Config(), device=DEVICE)
    raise ValueError(f"Unknown SGC algo: {algo!r}")


def _make_mas(algo: str, env: HESSMicrogridEnv) -> MADDPGSystem:
    obs_dim = env.obs_size_ess()
    obs_dims    = [obs_dim, obs_dim, obs_dim]
    action_dims = [1, 1, 1]
    base_algo = algo.replace("ma", "").replace("matd3", "td3").replace("mad3pg", "d3pg").replace("maddpg", "ddpg")
    cfg = TD3Config() if "td3" in algo else D3PGConfig() if "d3pg" in algo else DDPGConfig()
    return MADDPGSystem(obs_dims, action_dims, algorithm=base_algo, cfg=cfg, device=DEVICE)


def _savings_from_r_sum(r_sum: float, n_agents: int) -> float:
    """Reverse the normalisation from Eq. 27 to get £ savings."""
    return r_sum / (0.01 * n_agents)


# -------------------------------------------------------------------------
# Training loops
# -------------------------------------------------------------------------

def _run_sgc(
    algo: str,
    env: HESSMicrogridEnv,
    sim_cfg: SimConfig,
    verbose: bool,
) -> Dict[str, Any]:
    """Train a single global controller on case study 1."""
    obs_dim = env.obs_size_ess()
    agent = _make_sgc(algo, obs_dim, n_actions=3)

    global_step = 0
    episode_savings: List[float] = []
    eval_savings:    List[float] = []
    ess_losses:      List[float] = []

    for episode in range(sim_cfg.total_episodes):
        obs_list = env.reset()
        obs = obs_list[0]
        total_r_sum = 0.0
        total_ess_cost = 0.0

        for _ in range(sim_cfg.steps_per_episode):
            global_step += 1
            if global_step < sim_cfg.random_action_steps:
                action = np.random.uniform(-1, 1, size=3)
            else:
                action = agent.select_action(obs)

            next_obs_list, rewards, done, info = env.step(action.tolist())
            next_obs = next_obs_list[0]
            reward   = rewards[0]

            agent.push(obs, action, reward, next_obs, float(done))
            if global_step >= sim_cfg.learning_start_steps:
                agent.update()

            obs = next_obs
            total_r_sum   += info["R_sum"]
            total_ess_cost += info["R_CPC"] + info["R_SDC"]
            if done:
                break

        savings_raw = _savings_from_r_sum(total_r_sum, env.n_agents)
        savings_adj = savings_raw - total_ess_cost
        episode_savings.append(savings_raw)

        if episode >= sim_cfg.eval_start_episode:
            eval_savings.append(savings_raw)
            ess_losses.append(total_ess_cost)

        if verbose and episode % 20 == 0:
            m10 = np.mean(episode_savings[-10:])
            print(f"  [{algo.upper()} SGC ep {episode:3d}] "
                  f"savings=£{savings_raw:,.0f}  mean10=£{m10:,.0f}/week")

    return _compile(algo, eval_savings, ess_losses)


def _run_mas(
    algo: str,
    env: HESSMicrogridEnv,
    sim_cfg: SimConfig,
    verbose: bool,
) -> Dict[str, Any]:
    """Train a multi-agent MADDPG system on case study 1."""
    system = _make_mas(algo, env)

    global_step = 0
    episode_savings: List[float] = []
    eval_savings:    List[float] = []
    ess_losses:      List[float] = []

    for episode in range(sim_cfg.total_episodes):
        obs_list = env.reset()
        total_r_sum   = 0.0
        total_ess_cost = 0.0

        for _ in range(sim_cfg.steps_per_episode):
            global_step += 1
            if global_step < sim_cfg.random_action_steps:
                actions = [np.random.uniform(-1, 1, size=1) for _ in range(3)]
            else:
                actions = system.select_actions(obs_list)

            flat = [float(a) if np.ndim(a) == 0 else float(a[0]) for a in actions]
            next_obs_list, rewards, done, info = env.step(flat)

            system.push(obs_list, actions, rewards, next_obs_list, done)
            if global_step >= sim_cfg.learning_start_steps:
                system.update()

            obs_list = next_obs_list
            total_r_sum    += info["R_sum"]
            total_ess_cost += info["R_CPC"] + info["R_SDC"]
            if done:
                break

        savings_raw = _savings_from_r_sum(total_r_sum, env.n_agents)
        episode_savings.append(savings_raw)

        if episode >= sim_cfg.eval_start_episode:
            eval_savings.append(savings_raw)
            ess_losses.append(total_ess_cost)

        if verbose and episode % 20 == 0:
            m10 = np.mean(episode_savings[-10:])
            print(f"  [{algo.upper()} ep {episode:3d}] "
                  f"savings=£{savings_raw:,.0f}  mean10=£{m10:,.0f}/week")

    return _compile(algo, eval_savings, ess_losses)


def _run_madqn(
    env: HESSMicrogridEnv,
    sim_cfg: SimConfig,
    verbose: bool,
) -> Dict[str, Any]:
    """Train MADQN benchmark (Section 4.4.1)."""
    obs_dim = env.obs_size_ess()
    system = MADQNSystem(obs_dim, n_agents=3, n_actions=5, device=DEVICE)

    global_step = 0
    episode_savings: List[float] = []
    eval_savings:    List[float] = []
    ess_losses:      List[float] = []

    for episode in range(sim_cfg.total_episodes):
        obs_list = env.reset()
        total_r_sum   = 0.0
        total_ess_cost = 0.0

        for _ in range(sim_cfg.steps_per_episode):
            global_step += 1
            if global_step < sim_cfg.random_action_steps:
                actions = list(np.random.uniform(-1, 1, size=3))
            else:
                actions = system.select_actions(obs_list[:3])

            next_obs_list, rewards, done, info = env.step(actions)
            system.push(obs_list[:3], actions, rewards[:3], next_obs_list[:3], done)

            if global_step >= sim_cfg.learning_start_steps:
                system.update()

            obs_list = next_obs_list
            total_r_sum    += info["R_sum"]
            total_ess_cost += info["R_CPC"] + info["R_SDC"]
            if done:
                break

        savings_raw = _savings_from_r_sum(total_r_sum, env.n_agents)
        episode_savings.append(savings_raw)

        if episode >= sim_cfg.eval_start_episode:
            eval_savings.append(savings_raw)
            ess_losses.append(total_ess_cost)

        if verbose and episode % 20 == 0:
            m10 = np.mean(episode_savings[-10:])
            print(f"  [MADQN ep {episode:3d}] savings=£{savings_raw:,.0f}  mean10=£{m10:,.0f}/week")

    return _compile("madqn", eval_savings, ess_losses)


def _run_rbm(
    env: HESSMicrogridEnv,
    sim_cfg: SimConfig,
    verbose: bool,
) -> Dict[str, Any]:
    """Run Rule-Based Model benchmark (Section 4.4.2) — no training."""
    from maddpg_microgrid.config import ESS_CONFIGS, PV_MAX_MW, WIND_CONFIG
    from maddpg_microgrid.environment import wind_power_mw

    agent = RBMAgent()
    episode_savings: List[float] = []
    eval_savings:    List[float] = []
    ess_losses:      List[float] = []

    for episode in range(sim_cfg.total_episodes):
        obs_list = env.reset()
        total_r_sum   = 0.0
        total_ess_cost = 0.0

        for _ in range(sim_cfg.steps_per_episode):
            # Read raw env state from ESS charges and latest data
            charges = [ess.charge for ess in env.ess]
            d = env._data
            pv_mw = (d["irradiance"] / 1000.0) * PV_MAX_MW
            wt_mw = wind_power_mw(d["wind_speed"]) * WIND_CONFIG.n_turbines

            actions = agent.select_actions(
                demand=d["demand"],
                pv_mw=pv_mw,
                wt_mw=wt_mw,
                ess_charges=charges,
            )
            _, _, done, info = env.step(actions)

            total_r_sum    += info["R_sum"]
            total_ess_cost += info["R_CPC"] + info["R_SDC"]
            if done:
                break

        savings_raw = _savings_from_r_sum(total_r_sum, env.n_agents)
        episode_savings.append(savings_raw)

        if episode >= sim_cfg.eval_start_episode:
            eval_savings.append(savings_raw)
            ess_losses.append(total_ess_cost)

        if verbose and episode % 20 == 0:
            print(f"  [RBM ep {episode:3d}] savings=£{savings_raw:,.0f}/week")

    return _compile("rbm", eval_savings, ess_losses)


def _compile(algo: str, eval_savings: List[float], ess_losses: List[float]) -> Dict[str, Any]:
    total_raw = sum(eval_savings)
    total_ess = sum(ess_losses)
    return {
        "algo": algo,
        "episode_savings": eval_savings,
        "total_savings_gbp": total_raw,
        "total_ess_loss_gbp": total_ess,
        "adjusted_savings_gbp": total_raw - total_ess,
        "ess_loss_pct": (total_ess / total_raw * 100) if total_raw > 0 else 0.0,
    }


# -------------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------------

def run_case_study_1(
    algo: str = "maddpg",
    total_episodes: int = 200,
    seed: int = 42,
    verbose: bool = True,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Run Case Study 1 for one algorithm.

    Parameters
    ----------
    algo           : ddpg | d3pg | td3 | maddpg | mad3pg | matd3 | madqn | rbm
    total_episodes : 200 (paper default)
    seed           : random seed
    verbose        : print per-episode progress
    save           : save results JSON to results/

    Returns
    -------
    dict with total_savings_gbp, adjusted_savings_gbp, ess_loss_pct, episode_savings
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    algo = algo.lower()
    env = _env(seed, total_episodes)
    sim_cfg = env.sim_cfg   # use scaled sim_cfg from env

    print(f"\n{'='*60}")
    print(f"Case Study 1 — HESS-only ESS Control")
    print(f"Algorithm  : {algo.upper()}")
    print(f"Episodes   : {total_episodes}  (eval from ep {sim_cfg.eval_start_episode})")
    print(f"Device     : {DEVICE}")
    print(f"{'='*60}")

    t0 = time.time()

    if algo in ("ddpg", "d3pg", "td3"):
        results = _run_sgc(algo, env, sim_cfg, verbose)
    elif algo in ("maddpg", "mad3pg", "matd3"):
        results = _run_mas(algo, env, sim_cfg, verbose)
    elif algo == "madqn":
        results = _run_madqn(env, sim_cfg, verbose)
    elif algo == "rbm":
        results = _run_rbm(env, sim_cfg, verbose)
    else:
        raise ValueError(f"Unknown algorithm: {algo!r}")

    elapsed = time.time() - t0
    results["elapsed_s"] = elapsed
    results["seed"] = seed

    print(f"\n{'─'*60}")
    print(f"  Total savings (eval):   £{results['total_savings_gbp']:>10,.0f}")
    print(f"  ESS loss:               £{results['total_ess_loss_gbp']:>10,.0f}  "
          f"({results['ess_loss_pct']:.1f} %)")
    print(f"  Adjusted savings:       £{results['adjusted_savings_gbp']:>10,.0f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'─'*60}\n")

    if save:
        path = os.path.join(sim_config.RESULTS_DIR, f"cs1_{algo}_ep{total_episodes}.json")
        with open(path, "w") as fh:
            json.dump({k: (v if not isinstance(v, list) else v) for k, v in results.items()}, fh, indent=2)
        print(f"  Results saved → {path}")

    return results


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Case Study 1: HESS-only ESS control")
    parser.add_argument("--algo",     type=str, default="maddpg",
                        choices=["ddpg","d3pg","td3","maddpg","mad3pg","matd3","madqn","rbm"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--quiet",    action="store_true")
    parser.add_argument("--no-save",  action="store_true")
    args = parser.parse_args()

    run_case_study_1(
        algo=args.algo,
        total_episodes=args.episodes,
        seed=args.seed,
        verbose=not args.quiet,
        save=not args.no_save,
    )
