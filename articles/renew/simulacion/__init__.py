"""
Simulacion — Replication of the MADDPG microgrid paper
=======================================================
Source:
  "Multi-Agent Deep Deterministic Policy Gradient for Mixed Cooperative-Competitive
   Microgrid Energy Management with Renewable Energy Integration"

Directory layout
----------------
  benchmarks.py          – Rule-Based Model (RBM) and MADQN benchmark agents
  case_study_1.py        – Case Study 1: HESS-only ESS control
  case_study_2.py        – Case Study 2: HESS + MGA + 5 xMG energy trading
  compare_algorithms.py  – Run every algorithm and produce Table 3 / Table 4 comparison
  visualize.py           – Matplotlib plots (Fig 5 equivalents + savings table)
  run_simulation.py      – Main CLI entry point

Quick start
-----------
    python run_simulation.py --case 1 --algo maddpg --episodes 50
    python run_simulation.py --compare --case 1 --episodes 50
"""
