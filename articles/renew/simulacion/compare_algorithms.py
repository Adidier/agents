"""
Algorithm Comparison — Table 3 / Table 4 replication
=======================================================
Runs every algorithm on the selected case study and prints a
formatted comparison table matching the paper's Table 3 and Table 4.

Case Study 1 algorithms: DDPG, D3PG, TD3, MADDPG, MAD3PG, MATD3, MADQN, RBM
Case Study 2 algorithms: DDPG, D3PG, TD3, MADDPG, MAD3PG, MATD3

Results are also saved as JSON and a CSV summary.

Usage
-----
    # All algorithms, case study 1, 50 quick episodes
    python compare_algorithms.py --case 1 --episodes 50

    # Specific algorithms only
    python compare_algorithms.py --case 1 --algos ddpg maddpg rbm --episodes 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# path bootstrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_config  # noqa: F401

from sim_config import (
    SINGLE_AGENT_ALGOS, MULTI_AGENT_ALGOS, BENCHMARK_ALGOS, ALGO_LABELS, RESULTS_DIR
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _pct_vs_ddpg(val: float, ddpg_val: float) -> str:
    if ddpg_val == 0:
        return "—"
    return f"{(val - ddpg_val) / abs(ddpg_val) * 100:+.1f}%"


def _print_table(rows: List[Dict[str, Any]], title: str):
    """Print a formatted results table."""
    w = 80
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")
    header = (
        f"{'Algorithm':<12} {'Savings(£k)':>12} {'Adj(£k)':>10} "
        f"{'vs DDPG':>10} {'ESS Loss%':>10}"
    )
    print(header)
    print(f"{'─'*w}")

    ddpg_row = next((r for r in rows if r["algo"] == "ddpg"), None)
    ddpg_val = ddpg_row["total_savings_gbp"] / 1000 if ddpg_row else 1.0

    # Sort: single-agent first, then multi-agent, then benchmarks
    order = SINGLE_AGENT_ALGOS + MULTI_AGENT_ALGOS + BENCHMARK_ALGOS
    rows_sorted = sorted(rows, key=lambda r: (order.index(r["algo"]) if r["algo"] in order else 99))

    for r in rows_sorted:
        label = ALGO_LABELS.get(r["algo"], r["algo"].upper())
        sav   = r["total_savings_gbp"] / 1000
        adj   = r["adjusted_savings_gbp"] / 1000
        ess   = r["ess_loss_pct"]
        vs    = _pct_vs_ddpg(sav, ddpg_val)
        print(f"  {label:<10} {sav:>12,.2f} {adj:>10,.2f} {vs:>10} {ess:>9.1f}%")

    print(f"{'='*w}\n")


# -------------------------------------------------------------------------
# Run all / selected algorithms
# -------------------------------------------------------------------------

def compare_algorithms(
    case_study: int = 1,
    algos: Optional[List[str]] = None,
    total_episodes: int = 200,
    seed: int = 42,
    verbose: bool = False,
    save: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run selected algorithms on the given case study and return results.

    Parameters
    ----------
    case_study     : 1 or 2
    algos          : list of algorithm names; None = all applicable
    total_episodes : 200 (paper uses 200 with 100 eval)
    seed           : random seed
    verbose        : show per-episode output during training
    save           : persist results to results/

    Returns
    -------
    List of result dicts, one per algorithm.
    """
    if case_study == 1:
        default_algos = SINGLE_AGENT_ALGOS + MULTI_AGENT_ALGOS + BENCHMARK_ALGOS
    else:
        default_algos = SINGLE_AGENT_ALGOS + MULTI_AGENT_ALGOS

    algos_to_run = [a.lower() for a in (algos or default_algos)]

    # Validate
    if case_study == 2:
        unsupported = [a for a in algos_to_run if a in BENCHMARK_ALGOS]
        if unsupported:
            print(f"  [WARNING] Case Study 2 does not support benchmarks "
                  f"{unsupported}; skipping.")
            algos_to_run = [a for a in algos_to_run if a not in BENCHMARK_ALGOS]

    print(f"\n{'#'*70}")
    print(f"  ALGORITHM COMPARISON — Case Study {case_study}")
    print(f"  Algorithms : {', '.join(a.upper() for a in algos_to_run)}")
    print(f"  Episodes   : {total_episodes}")
    print(f"  Seed       : {seed}")
    print(f"{'#'*70}\n")

    all_results: List[Dict[str, Any]] = []

    for algo in algos_to_run:
        print(f"\n{'─'*50}")
        print(f"  Starting: {algo.upper()}")
        print(f"{'─'*50}")

        t0 = time.time()
        try:
            if case_study == 1:
                from case_study_1 import run_case_study_1
                res = run_case_study_1(algo, total_episodes=total_episodes,
                                       seed=seed, verbose=verbose, save=save)
            else:
                from case_study_2 import run_case_study_2
                res = run_case_study_2(algo, total_episodes=total_episodes,
                                       seed=seed, verbose=verbose, save=save)
            all_results.append(res)
        except Exception as exc:
            print(f"  [ERROR] {algo.upper()} failed: {exc}")
            import traceback
            traceback.print_exc()

        elapsed = time.time() - t0
        print(f"  {algo.upper()} done in {elapsed:.1f}s")

    if all_results:
        title = f"Case Study {case_study} — {total_episodes} episodes"
        _print_table(all_results, title)

        if save:
            summary_path = os.path.join(
                RESULTS_DIR, f"comparison_cs{case_study}_ep{total_episodes}.json"
            )
            with open(summary_path, "w") as fh:
                # Convert numpy types for JSON serialisation
                serialisable = []
                for r in all_results:
                    row = {}
                    for k, v in r.items():
                        if isinstance(v, list):
                            row[k] = [float(x) for x in v]
                        elif isinstance(v, (np.floating, np.integer)):
                            row[k] = float(v)
                        else:
                            row[k] = v
                    serialisable.append(row)
                json.dump(serialisable, fh, indent=2)
            print(f"  Full results saved → {summary_path}")

            # CSV summary (matches Table 3 structure)
            csv_path = os.path.join(
                RESULTS_DIR, f"comparison_cs{case_study}_ep{total_episodes}.csv"
            )
            with open(csv_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=[
                    "algo", "total_savings_gbp", "adjusted_savings_gbp",
                    "ess_loss_pct", "elapsed_s",
                ])
                writer.writeheader()
                for r in all_results:
                    writer.writerow({
                        "algo": r["algo"],
                        "total_savings_gbp": round(r["total_savings_gbp"], 2),
                        "adjusted_savings_gbp": round(r["adjusted_savings_gbp"], 2),
                        "ess_loss_pct": round(r["ess_loss_pct"], 2),
                        "elapsed_s": round(r.get("elapsed_s", 0), 1),
                    })
            print(f"  CSV summary saved   → {csv_path}")

    return all_results


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare all algorithms on one case study (Table 3 replication)"
    )
    parser.add_argument("--case",     type=int, default=1, choices=[1, 2])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--algos",    nargs="+", default=None,
                        help="Subset of algorithms to run (default: all)")
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--no-save",  action="store_true")
    args = parser.parse_args()

    compare_algorithms(
        case_study=args.case,
        algos=args.algos,
        total_episodes=args.episodes,
        seed=args.seed,
        verbose=args.verbose,
        save=not args.no_save,
    )
