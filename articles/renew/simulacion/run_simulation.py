"""
run_simulation.py — Main Entry Point
======================================
Replication of the MADDPG microgrid paper experiments.

Modes
-----
  Single algorithm:
    python run_simulation.py --case 1 --algo maddpg --episodes 200

  Full comparison (all algorithms, like Table 3):
    python run_simulation.py --compare --case 1 --episodes 200

  Quick smoke test (5 episodes each):
    python run_simulation.py --compare --case 1 --episodes 5

  Visualise saved results:
    python run_simulation.py --visualize results/comparison_cs1_ep200.json --case 1

Paper reference
---------------
  Case Study 1  : HESS-only ESS control (Section 5.1, Table 3 top half)
  Case Study 2  : HESS + MGA + 5 xMG energy trading (Section 5.2)
  Algorithms    : DDPG, D3PG, TD3  (single-agent)
                  MADDPG, MAD3PG, MATD3  (multi-agent)
                  MADQN, RBM  (benchmarks, CS1 only)
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the simulacion directory is on sys.path so local modules resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import sim_config  # noqa: F401  — also adds repo root to sys.path

from sim_config import (
    SINGLE_AGENT_ALGOS, MULTI_AGENT_ALGOS, BENCHMARK_ALGOS,
    ALGO_LABELS, DEFAULT_EPISODES, DEFAULT_SEED,
)


def _banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        MADDPG Microgrid — Energy Management Simulation       ║
║   Mixed Cooperative-Competitive MAS for RES Integration      ║
║              Articles / renew / simulacion                   ║
╚══════════════════════════════════════════════════════════════╝
    """)


def _run_single(args):
    if args.case == 1:
        from case_study_1 import run_case_study_1
        run_case_study_1(
            algo=args.algo,
            total_episodes=args.episodes,
            seed=args.seed,
            verbose=not args.quiet,
            save=not args.no_save,
        )
    else:
        from case_study_2 import run_case_study_2
        run_case_study_2(
            algo=args.algo,
            total_episodes=args.episodes,
            seed=args.seed,
            verbose=not args.quiet,
            save=not args.no_save,
        )


def _run_compare(args):
    from compare_algorithms import compare_algorithms

    algos = args.algos if args.algos else None
    results = compare_algorithms(
        case_study=args.case,
        algos=algos,
        total_episodes=args.episodes,
        seed=args.seed,
        verbose=not args.quiet,
        save=not args.no_save,
    )

    if results and not args.no_plot:
        from visualize import plot_comparison
        plot_comparison(results, case_study=args.case, show=False)


def _run_visualize(args):
    from visualize import load_results, plot_comparison
    results = load_results(args.visualize)
    plot_comparison(results, case_study=args.case, show=args.show_plot)


def main():
    _banner()

    all_algos_cs1 = SINGLE_AGENT_ALGOS + MULTI_AGENT_ALGOS + BENCHMARK_ALGOS
    all_algos_cs2 = SINGLE_AGENT_ALGOS + MULTI_AGENT_ALGOS

    parser = argparse.ArgumentParser(
        description="MADDPG microgrid simulation — paper replication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quickest possible test (5 eps):
    python run_simulation.py --compare --case 1 --episodes 5

  Full paper replication CS1 (takes several minutes):
    python run_simulation.py --compare --case 1 --episodes 200

  Single algorithm CS2:
    python run_simulation.py --case 2 --algo maddpg --episodes 200

  Visualise saved results:
    python run_simulation.py --visualize results/comparison_cs1_ep200.json --case 1
        """
    )

    # Mode flags
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--compare", action="store_true",
        help="Run all (or selected) algorithms and compare results (Table 3)"
    )
    mode_group.add_argument(
        "--visualize", type=str, metavar="JSON_PATH",
        help="Load a saved results JSON and generate plots only"
    )

    # Common options
    parser.add_argument("--case",     type=int, default=1, choices=[1, 2],
                        help="Case study (1=HESS only, 2=HESS+MGA+xMG)")
    parser.add_argument("--algo",     type=str, default="maddpg",
                        help="Algorithm to run in single mode "
                             f"(CS1: {all_algos_cs1}, CS2: {all_algos_cs2})")
    parser.add_argument("--algos",    nargs="+", default=None,
                        help="Subset of algorithms for --compare mode")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help=f"Total episodes (default={DEFAULT_EPISODES}; "
                             "paper uses 200, evaluation starts at ep 100)")
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress per-episode output")
    parser.add_argument("--no-save",  action="store_true",
                        help="Do not save result files")
    parser.add_argument("--no-plot",  action="store_true",
                        help="Skip plot generation after --compare")
    parser.add_argument("--show-plot", action="store_true",
                        help="Show interactive matplotlib window (requires display)")

    args = parser.parse_args()

    if args.visualize:
        _run_visualize(args)
    elif args.compare:
        _run_compare(args)
    else:
        _run_single(args)


if __name__ == "__main__":
    main()
