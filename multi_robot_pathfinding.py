from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot, inf
from pathlib import Path
import random
import time

import matplotlib.pyplot as plt
import pandas as pd


# 1. Configuration & Styling
DEFAULT_TRIALS_PER_CONFIGURATION = 5
RANDOM_SEED = 42
MIN_START_GOAL_DISTANCE = 6
MAX_HIGH_LEVEL_EXPANSIONS = 200

CONFIGURATIONS = [
    (rows, cols, obstacle_probability, number_of_agents)
    for rows, cols in [(15, 15), (20, 20)]
    for obstacle_probability in [0.15, 0.25]
    for number_of_agents in [3, 5]
]

OUTPUT_DIR = Path("multi_robot_results")
OUTPUT_DIR.mkdir(exist_ok=True)

class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

AGENT_COLORS = [
    Style.GREEN, Style.BLUE, Style.MAGENTA, Style.YELLOW,
    Style.CYAN, Style.RED, Style.WHITE, "\033[33m"
]

def print_header(title: str, width: int = 78):
    print(f"\n{Style.CYAN}╔{'═' * (width - 2)}╗{Style.RESET}")
    print(f"{Style.CYAN}║{Style.BOLD}{Style.WHITE} {title.center(width - 4)} {Style.RESET}{Style.CYAN}║{Style.RESET}")
    print(f"{Style.CYAN}╚{'═' * (width - 2)}╝{Style.RESET}")

def print_card(title: str, items: list[tuple[str, str]]):
    print(f"\n{Style.BOLD}{Style.MAGENTA}▸ {title}{Style.RESET}")
    print(f"{Style.DIM}{'─' * 54}{Style.RESET}")
    for label, val in items:
        print(f"  {Style.BOLD}{label:<28}{Style.RESET} : {Style.CYAN}{val}{Style.RESET}")

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WAIT = (0, 0)
SPACE_TIME_MOVES = MOVES + [WAIT]

@dataclass(frozen=True)
class AgentScenario:
    start: tuple[int, int]
    goal: tuple[int, int]

@dataclass
class AgentConstraints:
    vertex: set = field(default_factory=set)
    edge: set = field(default_factory=set)

    def copy(self):
        return AgentConstraints(set(self.vertex), set(self.edge))

# 2. Heuristics
class Heuristics:
    @staticmethod
    def manhattan(current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    @staticmethod
    def euclidean(current, goal):
        return hypot(current[0] - goal[0], current[1] - goal[1])

# 3. GridWorld
class GridWorld:
    def __init__(self, rows: int, cols: int, obstacle_prob: float, seed: int):
        self.rows = rows
        self.cols = cols
        self.obstacle_prob = obstacle_prob
        self.rng = random.Random(seed)
        self.grid = None
        self.agents = []

    def generate_grid(self):
        while True:
            grid = [
                [self.rng.random() < self.obstacle_prob for _ in range(self.cols)]
                for _ in range(self.rows)
            ]
            free_cells = sum(not cell for row in grid for cell in row)
            if free_cells >= 2:
                self.grid = grid
                return

    def get_reachable_cells(self, start):
        queue = deque([start])
        visited = {start}
        while queue:
            r, c = queue.popleft()
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                    and not self.grid[nr][nc] and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return list(visited)

    def generate_scenario(self, number_of_agents: int):
        while True:
            self.generate_grid()
            free_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) if not self.grid[r][c]]
            if len(free_cells) < 2 * number_of_agents:
                continue

            self.rng.shuffle(free_cells)
            used = set()
            self.agents = []

            for start in free_cells:
                if start in used:
                    continue
                reachable = self.get_reachable_cells(start)
                candidates = [
                    cell for cell in reachable
                    if cell not in used and abs(cell[0] - start[0]) + abs(cell[1] - start[1]) >= MIN_START_GOAL_DISTANCE
                ]
                if not candidates:
                    continue

                goal = self.rng.choice(candidates)
                self.agents.append(AgentScenario(start, goal))
                used.add(start)
                used.add(goal)

                if len(self.agents) == number_of_agents:
                    return self.grid, self.agents

    def local_obstacle_density(self, cell, radius=1):
        r, c = cell
        total = 0
        blocked = 0
        for rr in range(max(0, r - radius), min(self.rows, r + radius + 1)):
            for cc in range(max(0, c - radius), min(self.cols, c + radius + 1)):
                total += 1
                blocked += int(self.grid[rr][cc])
        return blocked / total if total else 0.0

    def render(self, paths=None, title="PATH OVERLAY"):
        starts = {agent.start: i for i, agent in enumerate(self.agents)}
        goals = {agent.goal: i for i, agent in enumerate(self.agents)}
        
        agent_path_cells = {}
        if paths:
            for i, path in enumerate(paths):
                for cell in path:
                    if cell not in starts and cell not in goals:
                        agent_path_cells[cell] = i

        print(f"\n  {Style.BOLD}{title}{Style.RESET}")
        print(f"  {Style.CYAN}┌" + "──" * self.cols + "─┐" + Style.RESET)

        for r in range(self.rows):
            line = [f"  {Style.CYAN}│{Style.RESET} "]
            for c in range(self.cols):
                cell = (r, c)
                if cell in starts:
                    i = starts[cell]
                    color = AGENT_COLORS[i % len(AGENT_COLORS)]
                    line.append(f"{Style.BOLD}{color}A{i % 10}{Style.RESET}")
                elif cell in goals:
                    i = goals[cell]
                    color = AGENT_COLORS[i % len(AGENT_COLORS)]
                    line.append(f"{Style.BOLD}{color}G{i % 10}{Style.RESET}")
                elif self.grid[r][c]:
                    line.append(f"{Style.DIM}# {Style.RESET}")
                elif cell in agent_path_cells:
                    i = agent_path_cells[cell]
                    color = AGENT_COLORS[i % len(AGENT_COLORS)]
                    line.append(f"{color}* {Style.RESET}")
                else:
                    line.append(f"{Style.DIM}· {Style.RESET}")
            line.append(f"{Style.CYAN}│{Style.RESET}")
            print("".join(line))

        print(f"  {Style.CYAN}└" + "──" * self.cols + "─┘" + Style.RESET)

# 4. Solvers
class MultiAgentSolver:
    def __init__(self, grid_world: GridWorld):
        self.world = grid_world

    def _reconstruct_path(self, parent, goal):
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path

    def astar_single(self, start, goal):
        open_heap = []
        insertion_counter = 0
        g_score = {start: 0}
        parent = {start: None}
        closed = set()

        heappush(open_heap, (Heuristics.manhattan(start, goal), 0, insertion_counter, start))
        nodes_expanded = 0

        while open_heap:
            _, negative_g, _, current = heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            nodes_expanded += 1

            if current == goal:
                path = self._reconstruct_path(parent, goal)
                return {"found": True, "path": path, "cost": len(path) - 1, "nodes_expanded": nodes_expanded}

            current_g = g_score[current]
            r, c = current
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.world.rows and 0 <= nc < self.world.cols) or self.world.grid[nr][nc]:
                    continue

                neighbor = (nr, nc)
                tentative_g = current_g + 1
                if tentative_g < g_score.get(neighbor, inf):
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    insertion_counter += 1
                    f = tentative_g + Heuristics.manhattan(neighbor, goal)
                    heappush(open_heap, (f, -tentative_g, insertion_counter, neighbor))

        return {"found": False, "path": None, "cost": inf, "nodes_expanded": nodes_expanded}

    def position_at(self, path, t):
        return path[t] if t < len(path) else path[-1]

    def detect_collisions(self, paths):
        if not paths:
            return {"collision_events": 0, "vertex_collisions": 0, "edge_swap_collisions": 0, "collision_pairs": 0}

        makespan = max(len(path) - 1 for path in paths)
        vertex_collisions = 0
        edge_swap_collisions = 0
        collision_pairs = set()

        for t in range(makespan + 1):
            occupied = defaultdict(list)
            for i, path in enumerate(paths):
                occupied[self.position_at(path, t)].append(i)

            for ids in occupied.values():
                if len(ids) > 1:
                    events = len(ids) * (len(ids) - 1) // 2
                    vertex_collisions += events
                    for a in range(len(ids)):
                        for b in range(a + 1, len(ids)):
                            collision_pairs.add(tuple(sorted((ids[a], ids[b]))))

            if t == makespan:
                continue

            transitions = {i: (self.position_at(path, t), self.position_at(path, t + 1)) for i, path in enumerate(paths)}
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    a0, a1 = transitions[i]
                    b0, b1 = transitions[j]
                    if a0 == b1 and a1 == b0 and a0 != a1:
                        edge_swap_collisions += 1
                        collision_pairs.add((i, j))

        return {
            "collision_events": vertex_collisions + edge_swap_collisions,
            "vertex_collisions": vertex_collisions,
            "edge_swap_collisions": edge_swap_collisions,
            "collision_pairs": len(collision_pairs),
        }

    def independent_astar(self):
        start_time = time.perf_counter()
        paths = []
        total_nodes = 0
        total_path_cost = 0

        for agent in self.world.agents:
            result = self.astar_single(agent.start, agent.goal)
            if not result["found"]:
                return self._failed_result(start_time, total_nodes, 0)
            paths.append(result["path"])
            total_nodes += result["nodes_expanded"]
            total_path_cost += result["cost"]

        collision_stats = self.detect_collisions(paths)
        makespan = max(len(path) - 1 for path in paths)
        success = collision_stats["collision_events"] == 0

        return {
            "success": success,
            "paths": paths,
            "makespan": makespan,
            "sum_of_costs": total_path_cost,
            "total_nodes_expanded": total_nodes,
            "planning_time_ms": (time.perf_counter() - start_time) * 1000,
            "wait_steps": sum(1 for path in paths for a, b in zip(path, path[1:]) if a == b),
            "high_level_expansions": 0,
            **collision_stats,
        }

    def _transition(self, path, t):
        return self.position_at(path, t), self.position_at(path, t + 1)

    def find_adaptive_conflict(self, paths):
        makespan = max(len(path) - 1 for path in paths)
        candidates = []

        for t in range(makespan + 1):
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    pi = self.position_at(paths[i], t)
                    pj = self.position_at(paths[j], t)

                    if pi == pj:
                        severity = 10.0 + 4.0 * self.world.local_obstacle_density(pi) + 0.01 * (makespan - t)
                        candidates.append((severity, t, "vertex", i, j, pi, None, None))

                    if t < makespan:
                        ai, ai_next = self._transition(paths[i], t)
                        aj, aj_next = self._transition(paths[j], t)
                        if ai == aj_next and ai_next == aj and ai != ai_next:
                            severity = 6.0 + 4.0 * self.world.local_obstacle_density(ai_next) + 0.01 * (makespan - t)
                            candidates.append((severity, t, "edge", i, j, None, ai, ai_next))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0]

    def _state_respects_constraints(self, constraints, current, nxt, next_time):
        if (next_time, nxt) in constraints.vertex: return False
        if (next_time - 1, current, nxt) in constraints.edge: return False
        return True

    def _has_future_goal_constraint(self, constraints, goal, arrival_time, horizon):
        for t in range(arrival_time + 1, horizon + 1):
            if (t, goal) in constraints.vertex: return True
        return False

    def low_level_constrained_astar(self, start, goal, constraints, horizon):
        open_heap = []
        insertion_counter = 0
        start_state = (start[0], start[1], 0)
        g_score = {start_state: 0}
        parent = {start_state: None}
        closed = set()

        heappush(open_heap, (Heuristics.manhattan(start, goal), 0, insertion_counter, start_state))
        nodes_expanded = 0

        while open_heap:
            _, _, _, state = heappop(open_heap)
            if state in closed: continue
            closed.add(state)
            nodes_expanded += 1

            r, c, t = state
            current = (r, c)

            if current == goal and not self._has_future_goal_constraint(constraints, goal, t, horizon):
                path_states = self._reconstruct_path(parent, state)
                return [s[:2] for s in path_states], nodes_expanded

            if t >= horizon: continue

            current_g = g_score[state]

            for dr, dc in SPACE_TIME_MOVES:
                nr, nc = r + dr, c + dc
                next_time = t + 1
                if not (0 <= nr < self.world.rows and 0 <= nc < self.world.cols): continue
                if self.world.grid[nr][nc]: continue

                nxt = (nr, nc)
                if not self._state_respects_constraints(constraints, current, nxt, next_time): continue

                next_state = (nr, nc, next_time)
                tentative_g = current_g + 1
                if tentative_g < g_score.get(next_state, inf):
                    g_score[next_state] = tentative_g
                    parent[next_state] = state
                    insertion_counter += 1
                    h = Heuristics.manhattan(nxt, goal)
                    heappush(open_heap, (tentative_g + h, -tentative_g, insertion_counter, next_state))

        return None, nodes_expanded

    def _path_has_conflict(self, path, constraints, horizon):
        for t, cell in enumerate(path):
            if (t, cell) in constraints.vertex: return True
            if t > 0 and (t - 1, path[t - 1], cell) in constraints.edge: return True
        if self._has_future_goal_constraint(constraints, path[-1], len(path) - 1, horizon): return True
        return False

    def _failed_result(self, start_time, total_nodes, high_level_expansions):
        return {
            "success": False, "paths": None, "makespan": inf, "sum_of_costs": inf,
            "total_nodes_expanded": total_nodes, "planning_time_ms": (time.perf_counter() - start_time) * 1000,
            "collision_events": inf, "vertex_collisions": inf, "edge_swap_collisions": inf,
            "collision_pairs": inf, "wait_steps": inf, "high_level_expansions": high_level_expansions,
        }

    def cba_star(self):
        start_time = time.perf_counter()
        n_agents = len(self.world.agents)
        horizon = 0
        root_paths = []
        total_nodes = 0

        for agent in self.world.agents:
            result = self.astar_single(agent.start, agent.goal)
            if not result["found"]:
                return self._failed_result(start_time, total_nodes, 0)
            root_paths.append(result["path"])
            total_nodes += result["nodes_expanded"]
            horizon = max(horizon, result["cost"])

        horizon += max(20, self.world.rows + self.world.cols) + 5 * n_agents
        root_constraints = [AgentConstraints() for _ in range(n_agents)]
        root_conflicts = self.detect_collisions(root_paths)["collision_events"]

        open_heap = []
        root_cost = sum(len(path) - 1 for path in root_paths)
        heappush(open_heap, (root_cost + 10.0 * root_conflicts, root_cost, root_conflicts, 0, root_constraints, root_paths))
        expansions = 0

        while open_heap and expansions < MAX_HIGH_LEVEL_EXPANSIONS:
            _, total_cost, _, _, constraints, paths = heappop(open_heap)
            expansions += 1

            conflict = self.find_adaptive_conflict(paths)
            if conflict is None:
                collisions = self.detect_collisions(paths)
                return {
                    "success": True, "paths": paths, "makespan": max(len(path) - 1 for path in paths),
                    "sum_of_costs": total_cost, "total_nodes_expanded": total_nodes,
                    "planning_time_ms": (time.perf_counter() - start_time) * 1000,
                    "wait_steps": sum(1 for path in paths for a, b in zip(path, path[1:]) if a == b),
                    "high_level_expansions": expansions, **collisions,
                }

            _, conflict_time, conflict_type, i, j, cell, edge_from, edge_to = conflict

            for constrained_agent in (i, j):
                child_constraints = [c.copy() for c in constraints]
                if conflict_type == "vertex":
                    child_constraints[constrained_agent].vertex.add((conflict_time, cell))
                else:
                    if constrained_agent == i:
                        child_constraints[constrained_agent].edge.add((conflict_time, edge_from, edge_to))
                    else:
                        child_constraints[constrained_agent].edge.add((conflict_time, edge_to, edge_from))

                new_path, expanded = self.low_level_constrained_astar(
                    self.world.agents[constrained_agent].start,
                    self.world.agents[constrained_agent].goal,
                    child_constraints[constrained_agent],
                    horizon
                )
                total_nodes += expanded
                if not new_path or self._path_has_conflict(new_path, child_constraints[constrained_agent], horizon):
                    continue

                child_paths = list(paths)
                child_paths[constrained_agent] = new_path
                child_conflicts = self.detect_collisions(child_paths)["collision_events"]
                child_cost = sum(len(path) - 1 for path in child_paths)
                heappush(open_heap, (
                    child_cost + 10.0 * child_conflicts, child_cost, child_conflicts,
                    expansions * 10 + constrained_agent, child_constraints, child_paths
                ))

        return self._failed_result(start_time, total_nodes, expansions)


# 5. Graph and Report Generation
class ExperimentRunner:
    @staticmethod
    def run_experiments():
        records = []
        for config_id, (rows, cols, obs_prob, num_agents) in enumerate(CONFIGURATIONS, start=1):
            print(f"  {Style.CYAN}⚙️  Config {config_id}/{len(CONFIGURATIONS)}:{Style.RESET} {rows}×{cols} grid | {obs_prob:.0%} obstacles | {num_agents} agents...")
            for trial in range(1, DEFAULT_TRIALS_PER_CONFIGURATION + 1):
                world = GridWorld(rows, cols, obs_prob, seed=RANDOM_SEED + trial * 1000 + config_id)
                world.generate_scenario(num_agents)
                solver = MultiAgentSolver(world)

                baseline = solver.independent_astar()
                proposed = solver.cba_star()

                common = {
                    "configuration": config_id, "trial": trial, "rows": rows, "cols": cols,
                    "grid_size": f"{rows}x{cols}", "obstacle_probability": obs_prob, "number_of_agents": num_agents,
                }

                for name, result in [("Independent A*", baseline), ("CBA* (Proposed)", proposed)]:
                    record = common.copy()
                    record["method"] = name
                    record.update({k: v for k, v in result.items() if k != "paths"})
                    records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(OUTPUT_DIR / "multi_robot_results.csv", index=False)
        return df

    @staticmethod
    def summarize(results):
        group_cols = ["rows", "cols", "obstacle_probability", "number_of_agents", "method"]
        rows = []
        for keys, group in results.groupby(group_cols):
            success = group[group["success"] == True]
            row = dict(zip(group_cols, keys))
            row["trials"] = len(group)
            row["success_rate"] = 100 * group["success"].mean()
            row["avg_makespan_success"] = success["makespan"].mean() if len(success) else inf
            row["avg_sum_of_costs_success"] = success["sum_of_costs"].mean() if len(success) else inf
            row["avg_collisions"] = group["collision_events"].replace([inf, -inf], pd.NA).mean()
            row["avg_wait_steps_success"] = success["wait_steps"].mean() if len(success) else inf
            row["avg_nodes_expanded"] = group["total_nodes_expanded"].mean()
            row["avg_planning_time_ms"] = group["planning_time_ms"].mean()
            row["avg_high_level_expansions"] = group["high_level_expansions"].mean()
            rows.append(row)
        
        summary = pd.DataFrame(rows)
        summary.to_csv(OUTPUT_DIR / "multi_robot_summary.csv", index=False)
        
        overall = summary.groupby("method", as_index=False).agg({
            "success_rate": "mean", "avg_makespan_success": "mean", "avg_sum_of_costs_success": "mean",
            "avg_collisions": "mean", "avg_wait_steps_success": "mean", "avg_nodes_expanded": "mean",
            "avg_planning_time_ms": "mean", "avg_high_level_expansions": "mean",
        })
        overall.to_csv(OUTPUT_DIR / "multi_robot_overall_comparison.csv", index=False)
        return summary, overall

    @staticmethod
    def generate_plots(summary):
        def _plot(metric, ylabel, title, filename, groupby_col):
            plt.figure(figsize=(10, 6))
            for method in summary["method"].unique():
                data = summary[summary["method"] == method]
                grouped = data.groupby(groupby_col)[metric].mean()
                x_vals = grouped.index * 100 if groupby_col == "obstacle_probability" else grouped.index
                plt.plot(x_vals, grouped.values, marker="o", label=method)
            plt.xlabel(groupby_col.replace("_", " ").title())
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / filename, dpi=180)
            plt.close()

        _plot("success_rate", "Success rate (%)", "Success Rate vs Number of Agents", "success_rate_vs_agents.png", "number_of_agents")
        _plot("avg_makespan_success", "Average makespan", "Makespan vs Number of Agents", "makespan_vs_agents.png", "number_of_agents")
        _plot("avg_collisions", "Average collision events", "Collision Events vs Number of Agents", "collisions_vs_agents.png", "number_of_agents")
        _plot("avg_planning_time_ms", "Average planning time (ms)", "Planning Time vs Number of Agents", "planning_time_vs_agents.png", "number_of_agents")
        _plot("success_rate", "Success rate (%)", "Success Rate vs Obstacle Density", "success_rate_vs_obstacles.png", "obstacle_probability")
        _plot("avg_planning_time_ms", "Average planning time (ms)", "Planning Time vs Grid Size", "planning_time_vs_grid.png", "rows")

    @staticmethod
    def save_example():
        world = GridWorld(20, 30, 0.22, seed=1042)
        world.generate_scenario(5)
        solver = MultiAgentSolver(world)
        
        baseline = solver.independent_astar()
        proposed = solver.cba_star()

        if not proposed["paths"]:
            return None

        info = pd.DataFrame([
            {"agent": i, "start": a.start, "goal": a.goal,
             "independent_path_length": len(baseline["paths"][i]) - 1,
             "proposed_path_length": len(proposed["paths"][i]) - 1}
            for i, a in enumerate(world.agents)
        ])
        info.to_csv(OUTPUT_DIR / "example_agent_paths.csv", index=False)
        return world, baseline, proposed

# 6. Main
def main():
    print_header("🤖 MULTI-ROBOT PATH-FINDING: INDEPENDENT A* VS CBA* 🤖")
    print_card("Problem Configuration", [
        ("Benchmark Configurations", f"{len(CONFIGURATIONS)} setups"),
        ("Trials per Configuration", f"{DEFAULT_TRIALS_PER_CONFIGURATION}"),
        ("Max High-Level Expansions", f"{MAX_HIGH_LEVEL_EXPANSIONS}"),
        ("Min Start-Goal Distance", f"{MIN_START_GOAL_DISTANCE} cells"),
        ("Random Seed", f"{RANDOM_SEED}"),
        ("Output Folder", f"{OUTPUT_DIR}/"),
    ])

    print(f"\n{Style.YELLOW}⏳ Running multi-robot benchmark trials across environments...{Style.RESET}\n")

    results = ExperimentRunner.run_experiments()
    summary, overall = ExperimentRunner.summarize(results)
    ExperimentRunner.generate_plots(summary)

    example = ExperimentRunner.save_example()

    print_header("📊 BENCHMARK OVERALL COMPARISON")
    headers = ["Method", "Success", "Avg Makespan", "Sum of Costs", "Collisions", "Wait Steps", "Time (ms)"]
    top_line = "  ┌─{:─<18}─┬─{:─<9}─┬─{:─<14}─┬─{:─<14}─┬─{:─<12}─┬─{:─<12}─┬─{:─<11}─┐".format("", "", "", "", "", "", "")
    div_line = "  ├─{:─<18}─┼─{:─<9}─┼─{:─<14}─┼─{:─<14}─┼─{:─<12}─┼─{:─<12}─┼─{:─<11}─┤".format("", "", "", "", "", "", "")
    bot_line = "  └─{:─<18}─┴─{:─<9}─┴─{:─<14}─┴─{:─<14}─┴─{:─<12}─┴─{:─<12}─┴─{:─<11}─┘".format("", "", "", "", "", "", "")

    print(f"\n{top_line}")
    print(f"  │ {Style.BOLD}{headers[0]:<18}{Style.RESET} │ {Style.BOLD}{headers[1]:>9}{Style.RESET} │ {Style.BOLD}{headers[2]:>14}{Style.RESET} │ {Style.BOLD}{headers[3]:>14}{Style.RESET} │ {Style.BOLD}{headers[4]:>12}{Style.RESET} │ {Style.BOLD}{headers[5]:>12}{Style.RESET} │ {Style.BOLD}{headers[6]:>11}{Style.RESET} │")
    print(div_line)

    for _, row in overall.iterrows():
        method_name = row["method"]
        succ = f"{row['success_rate']:.1f}%"
        make = f"{row['avg_makespan_success']:.2f}"
        soc = f"{row['avg_sum_of_costs_success']:.2f}"
        coll = f"{row['avg_collisions']:.2f}"
        waits = f"{row['avg_wait_steps_success']:.2f}"
        runtime = f"{row['avg_planning_time_ms']:.2f}"

        color = Style.GREEN if "CBA*" in method_name else Style.YELLOW
        coll_color = Style.GREEN if coll == "0.00" else Style.RED
        print(f"  │ {color}{method_name:<18}{Style.RESET} │ {succ:>9} │ {make:>14} │ {soc:>14} │ {coll_color}{coll:>12}{Style.RESET} │ {waits:>12} │ {runtime:>11} │")

    print(bot_line)

    if example:
        world, baseline, proposed = example
        print_header("🗺️  SAMPLE MULTI-ROBOT MAP VISUALIZATION")
        
        solver = MultiAgentSolver(world)
        base_col = solver.detect_collisions(baseline["paths"])["collision_events"]
        prop_col = solver.detect_collisions(proposed["paths"])["collision_events"]

        world.render(baseline["paths"], f"1) Independent A* (Collisions: {base_col})")
        world.render(proposed["paths"], f"2) CBA* - Collision-Free (Collisions: {prop_col})")

        print(f"\n  {Style.BOLD}Agent Assignments:{Style.RESET}")
        for i, agent in enumerate(world.agents):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            base_len = len(baseline['paths'][i]) - 1 if baseline['paths'] else 0
            prop_len = len(proposed['paths'][i]) - 1 if proposed['paths'] else 0
            print(f"    {color}● Agent {i}{Style.RESET}: {agent.start} → {agent.goal} | Indep steps: {base_len} | CBA* steps: {prop_len}")

    print(f"\n  {Style.GREEN}✔ Plots and CSV summaries saved to: {OUTPUT_DIR.resolve()}{Style.RESET}\n")

if __name__ == "__main__":
    main()
