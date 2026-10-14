# 🤖 Robot Pathfinding Using A* & Multi-Agent Conflict-Based Adaptive A*

An Artificial Intelligence project exploring single-agent and multi-agent pathfinding on 2D grid environments with static obstacles and collision-avoidance constraints.

---

## 📌 Project Overview

This project implements and evaluates heuristic search algorithms for autonomous robot navigation:

1. **Single-Robot Pathfinding (`astar_robot_pathfinding.py`)**:
   - Navigates an agent from Start to Goal on a 2D grid while avoiding obstacles (4-connected grid).
   - Evaluates and benchmarks 5 different heuristic functions:
     - **Manhattan Distance** ($L_1$ norm — optimal & admissible for 4-directional grid movement)
     - **Euclidean Distance** ($L_2$ straight-line distance)
     - **Chebyshev Distance** ($L_\infty$ maximum coordinate difference)
     - **Diagonal (Octile) Distance**
     - **Zero Heuristic (Dijkstra)** (Uninformed baseline)
   - Evaluates performance across runtime, nodes expanded, path cost, and scalability.

2. **Multi-Robot Pathfinding (`multi_robot_pathfinding.py`)**:
   - Solves Multi-Agent Pathfinding (MAPF) for multiple robots (3–5+ agents) operating concurrently in a shared workspace.
   - Prevents both **vertex collisions** (robots occupying the same cell at time $t$) and **edge/swap collisions** (robots crossing each other between $t$ and $t+1$).
   - Compares:
     - **Independent A\***: Baseline where agents plan paths individually without multi-agent coordination.
     - **Conflict-Based Adaptive A\* (CBA\*)**: Priority- and reservation-based space-time planning with adaptive replanning to eliminate multi-agent collisions.
   - Measures makespan, sum of costs, planning duration, success rate, and collision counts.

---

## 📂 Repository Structure

- `astar_robot_pathfinding.py` — Single-robot A* search implementation and heuristic comparative benchmarks.
- `multi_robot_pathfinding.py` — Multi-agent path planning, conflict detection, and resolution algorithms.
- `Project_Report_And_Presentation_Guide.md` — Detailed technical documentation, theoretical background, and presentation defense guide.
- `Assignment.pdf` — Original assignment specification and problem requirements.
- `A-Robot-Pathfinding-and-Conflict-Based-Adaptive-A.pptx` — Presentation slide deck detailing findings and architecture.
- `astar_results/` — Generated benchmark graphs, runtime comparisons, and summary CSV reports for single-robot heuristics.
- `multi_robot_results/` — Comprehensive evaluation plots (success rates, collisions, makespan vs. agent count, planning times) and CSV summaries.
