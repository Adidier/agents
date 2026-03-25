"""
Visualisation
=============
Generates plots equivalent to the paper's figures using results produced
by compare_algorithms.py or case_study_1/2.py.

Figures produced:
  Fig 5(a) equiv  — Smoothed episodic reward curves per algorithm
  Fig 5(b) equiv  — Cumulative adjusted savings over evaluation episodes
  Table 3 bar     — Bar chart of total savings per algorithm
  ESS SoC trace   — Example ESS state-of-charge over one episode (for analysis)

Usage
-----
    # From JSON results file:
    python visualize.py --file results/comparison_cs1_ep200.json

    # After running compare_algorithms programmatically:
    from visualize import plot_comparison
    plot_comparison(results_list, case_study=1)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_config  # noqa: F401

from sim_config import ALGO_LABELS, ALGO_STYLES, RESULTS_DIR


# -------------------------------------------------------------------------
# Smoothing helper
# -------------------------------------------------------------------------

def _smooth(values: List[float], window: int = 10) -> List[float]:
    """Moving-average smoothing, matches paper's Fig 5 appearance."""
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


# -------------------------------------------------------------------------
# Plot: smoothed episodic rewards (Fig 5a equivalent)
# -------------------------------------------------------------------------

def plot_episodic_rewards(
    results_list: List[Dict[str, Any]],
    case_study: int = 1,
    smooth_window: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Line plot of smoothed weekly savings over evaluation episodes.
    Replicates Fig 5(a) from the paper.
    """
    try:
        import matplotlib
        matplotlib.use("Agg" if not show else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results_list:
        algo  = r["algo"]
        label = ALGO_LABELS.get(algo, algo.upper())
        style = ALGO_STYLES.get(algo, {})
        eps   = r.get("episode_savings", [])
        if not eps:
            continue
        smoothed = _smooth(eps, window=smooth_window)
        x = list(range(1, len(smoothed) + 1))
        ax.plot(x, [v / 1000 for v in smoothed],
                label=label,
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                linewidth=1.8)

    ax.set_xlabel("Evaluation Episode", fontsize=12)
    ax.set_ylabel("Weekly Savings (£k)", fontsize=12)
    ax.set_title(f"Case Study {case_study} — Smoothed Episodic Rewards "
                 f"(window={smooth_window})", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"fig5a_cs{case_study}_reward_curves.png")
    fig.savefig(save_path, dpi=150)
    print(f"  Plot saved → {save_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


# -------------------------------------------------------------------------
# Plot: cumulative adjusted savings (Fig 5b equivalent)
# -------------------------------------------------------------------------

def plot_cumulative_savings(
    results_list: List[Dict[str, Any]],
    case_study: int = 1,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Cumulative adjusted savings over evaluation episodes.
    Replicates Fig 5(b) from the paper.
    """
    try:
        import matplotlib
        matplotlib.use("Agg" if not show else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results_list:
        algo  = r["algo"]
        label = ALGO_LABELS.get(algo, algo.upper())
        style = ALGO_STYLES.get(algo, {})
        eps   = r.get("episode_savings", [])
        if not eps:
            continue
        cumulative = np.cumsum([v / 1000 for v in eps])
        x = list(range(1, len(cumulative) + 1))
        ax.plot(x, cumulative,
                label=label,
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                linewidth=1.8)

    ax.set_xlabel("Evaluation Episode", fontsize=12)
    ax.set_ylabel("Cumulative Savings (£k)", fontsize=12)
    ax.set_title(f"Case Study {case_study} — Cumulative Adjusted Savings", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"fig5b_cs{case_study}_cumulative_savings.png")
    fig.savefig(save_path, dpi=150)
    print(f"  Plot saved → {save_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


# -------------------------------------------------------------------------
# Plot: bar chart of total savings (Table 3 visualisation)
# -------------------------------------------------------------------------

def plot_savings_bar(
    results_list: List[Dict[str, Any]],
    case_study: int = 1,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Grouped bar chart: raw vs adjusted savings per algorithm.
    """
    try:
        import matplotlib
        matplotlib.use("Agg" if not show else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available; skipping plot.")
        return

    from sim_config import SINGLE_AGENT_ALGOS, MULTI_AGENT_ALGOS, BENCHMARK_ALGOS
    order = SINGLE_AGENT_ALGOS + MULTI_AGENT_ALGOS + BENCHMARK_ALGOS

    rows = sorted(results_list, key=lambda r: (order.index(r["algo"]) if r["algo"] in order else 99))

    labels = [ALGO_LABELS.get(r["algo"], r["algo"].upper()) for r in rows]
    raw    = [r["total_savings_gbp"] / 1000 for r in rows]
    adj    = [r["adjusted_savings_gbp"] / 1000 for r in rows]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - w/2, raw, w, label="Raw Savings",      color="#4878d0", alpha=0.85)
    bars2 = ax.bar(x + w/2, adj, w, label="Adjusted Savings", color="#ee854a", alpha=0.85)

    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Total Savings (£k)", fontsize=12)
    ax.set_title(f"Case Study {case_study} — Total Energy Savings Comparison", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"table3_cs{case_study}_savings_bar.png")
    fig.savefig(save_path, dpi=150)
    print(f"  Plot saved → {save_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


# -------------------------------------------------------------------------
# Combined comparison export
# -------------------------------------------------------------------------

def plot_comparison(
    results_list: List[Dict[str, Any]],
    case_study: int = 1,
    show: bool = False,
):
    """
    Generate all three comparison plots from a results list.
    Useful for programmatic use after compare_algorithms().
    """
    plot_episodic_rewards(results_list, case_study=case_study, show=show)
    plot_cumulative_savings(results_list, case_study=case_study, show=show)
    plot_savings_bar(results_list, case_study=case_study, show=show)
    print("  All plots saved to results/")


# -------------------------------------------------------------------------
# Load from file helper
# -------------------------------------------------------------------------

def load_results(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path) as fh:
        return json.load(fh)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise comparison results")
    parser.add_argument("--file", type=str, required=True,
                        help="Path to JSON results file from compare_algorithms.py")
    parser.add_argument("--case", type=int, default=1, choices=[1, 2])
    parser.add_argument("--show", action="store_true",
                        help="Display interactive plot window (requires display)")
    args = parser.parse_args()

    results = load_results(args.file)
    plot_comparison(results, case_study=args.case, show=args.show)
