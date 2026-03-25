"""
Simulation-level configuration
================================
Ties together the paper's experimental setup (Section 4.3) with
the RL-framework config objects from maddpg_microgrid/config.py.

Import this module first in every script to ensure sys.path is correct.
The local maddpg_microgrid/ package lives inside this same directory.
"""

import os
import sys

# Add the simulacion directory to sys.path so that:
#   import maddpg_microgrid   (local copy)
#   import benchmarks         (local module)
# both resolve correctly from any working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---- algorithm registry -----------------------------------------------
SINGLE_AGENT_ALGOS = ["ddpg", "d3pg", "td3"]
MULTI_AGENT_ALGOS  = ["maddpg", "mad3pg", "matd3"]
BENCHMARK_ALGOS    = ["madqn", "rbm"]

ALL_ALGOS = SINGLE_AGENT_ALGOS + MULTI_AGENT_ALGOS + BENCHMARK_ALGOS

# Pretty labels for tables / plots
ALGO_LABELS = {
    "ddpg":    "DDPG",
    "d3pg":    "D3PG",
    "td3":     "TD3",
    "maddpg":  "MADDPG",
    "mad3pg":  "MAD3PG",
    "matd3":   "MATD3",
    "madqn":   "MADQN",
    "rbm":     "RBM",
}

# Line styles for plots (matches paper where possible)
ALGO_STYLES = {
    "ddpg":   {"color": "#1f77b4", "linestyle": "--"},
    "d3pg":   {"color": "#ff7f0e", "linestyle": "--"},
    "td3":    {"color": "#2ca02c", "linestyle": "--"},
    "maddpg": {"color": "#d62728", "linestyle": "-"},
    "mad3pg": {"color": "#9467bd", "linestyle": "-"},
    "matd3":  {"color": "#8c564b", "linestyle": "-"},
    "madqn":  {"color": "#e377c2", "linestyle": "-."},
    "rbm":    {"color": "#7f7f7f", "linestyle": ":"},
}

# ---- simulation defaults ----------------------------------------------
DEFAULT_EPISODES: int = 200   # as per paper
DEFAULT_SEED:     int = 42
RESULTS_DIR: str = os.path.join(_HERE, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)
