from heapq import heappush, heappop
from collections import deque
from math import hypot, inf, sqrt
from pathlib import Path
import random
import time

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Style and Formatting
# ---------------------------------------------------------
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
    BG_DARK = "\033[48;5;236m"


def print_header(title):
    width = 76
    print(f"\n{Style.CYAN}╔{'═' * (width - 2)}╗{Style.RESET}")
    print(f"{Style.CYAN}║{Style.BOLD}{Style.WHITE} {title.center(width - 4)} {Style.RESET}{Style.CYAN}║{Style.RESET}")
    print(f"{Style.CYAN}╚{'═' * (width - 2)}╝{Style.RESET}")


def print_card(title, items):
    print(f"\n{Style.BOLD}{Style.MAGENTA}▸ {title}{Style.RESET}")
    print(f"{Style.DIM}{'─' * 50}{Style.RESET}")
    for label, val in items:
        print(f"  {Style.BOLD}{label:<26}{Style.RESET} : {Style.CYAN}{val}{Style.RESET}")


# ---------------------------------------------------------
# GridWorld Class
# ---------------------------------------------------------
class GridWorld:
    """Encapsulates grid generation, validation, and rendering."""
    MOVES = [
        (-1, 0),  # Up
        (1, 0),   # Down
        (0, -1),  # Left
        (0, 1),   # Right
    ]

    def __init__(self, rows=20, cols=20, obstacle_prob=0.15, seed=42):
        self.rows = rows
        self.cols = cols
        self.obstacle_prob = obstacle_prob
        self.rng = random.Random(seed)
        self.grid = None
        self.start = None
        self.goal = None

    def generate_grid(self):
        """Generate a random grid."""
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
        """Return all free cells reachable from start using BFS."""
        queue = deque([start])
        visited = {start}
        while queue:
            r, c = queue.popleft()
            for dr, dc in self.MOVES:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and not self.grid[nr][nc] and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return list(visited)

    def generate_solvable_scenario(self, min_distance=3):
        """Generate a guaranteed solvable start and goal configuration."""
        while True:
            self.generate_grid()
            free_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) if not self.grid[r][c]]
            start = self.rng.choice(free_cells)
            reachable = self.get_reachable_cells(start)
            candidates = [
                cell for cell in reachable
                if abs(cell[0] - start[0]) + abs(cell[1] - start[1]) >= min_distance
            ]
            if candidates:
                self.start = start
                self.goal = self.rng.choice(candidates)
                return self.start, self.goal

    def render(self, path=None):
        """Print an aesthetically styled grid in the terminal."""
        path_set = set(path or [])
        print(f"{Style.CYAN}┌" + "──" * self.cols + "─┐" + Style.RESET)
        for r in range(self.rows):
            line = [f"{Style.CYAN}│{Style.RESET} "]
            for c in range(self.cols):
                cell = (r, c)
                if cell == self.start:
                    line.append(f"{Style.BOLD}{Style.GREEN}S{Style.RESET} ")
                elif cell == self.goal:
                    line.append(f"{Style.BOLD}{Style.RED}G{Style.RESET} ")
                elif self.grid[r][c]:
                    line.append(f"{Style.DIM}# {Style.RESET}")
                elif cell in path_set:
                    line.append(f"{Style.BOLD}{Style.YELLOW}*{Style.RESET} ")
                else:
                    line.append(f"{Style.DIM}· {Style.RESET}")
            line.append(f"{Style.CYAN}│{Style.RESET}")
            print("".join(line))
        print(f"{Style.CYAN}└" + "──" * self.cols + "─┘" + Style.RESET)


# ---------------------------------------------------------
# Heuristics
# ---------------------------------------------------------
class Heuristics:
    """Dedicated heuristic functions."""
    @staticmethod
    def dijkstra(current, goal):
        return 0

    @staticmethod
    def manhattan(current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    @staticmethod
    def euclidean(current, goal):
        return hypot(current[0] - goal[0], current[1] - goal[1])

    @staticmethod
    def chebyshev(current, goal):
        return max(abs(current[0] - goal[0]), abs(current[1] - goal[1]))
    
    @staticmethod
    def octile(current, goal):
        dx = abs(current[0] - goal[0])
        dy = abs(current[1] - goal[1])
        return max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)


# ---------------------------------------------------------
# A* Solver Class
# ---------------------------------------------------------
class AStarSolver:
    """Encapsulates node expansion, heuristics, and frontier tracking."""
    def __init__(self, grid_world: GridWorld, heuristic_func):
        self.grid_world = grid_world
        self.heuristic_func = heuristic_func
        
    def _reconstruct_path(self, parent, current):
        path = []
        while current is not None:
            path.append(current)
            current = parent.get(current)
        path.reverse()
        return path

    def solve(self):
        """Run A* search."""
        start = self.grid_world.start
        goal = self.grid_world.goal
        grid = self.grid_world.grid
        rows = self.grid_world.rows
        cols = self.grid_world.cols
        moves = self.grid_world.MOVES

        start_time = time.perf_counter()

        open_heap = []
        insertion_counter = 0

        g_score = {start: 0}
        parent = {start: None}
        closed = set()

        initial_f = self.heuristic_func(start, goal)
        heappush(open_heap, (initial_f, 0, insertion_counter, start))

        nodes_expanded = 0
        max_frontier = 1

        while open_heap:
            f_score, negative_g, _, current = heappop(open_heap)
            current_g = g_score.get(current, inf)

            if current in closed:
                continue

            closed.add(current)
            nodes_expanded += 1

            if current == goal:
                path = self._reconstruct_path(parent, goal)
                return {
                    "found": True,
                    "path": path,
                    "path_cost": len(path) - 1,
                    "nodes_expanded": nodes_expanded,
                    "max_frontier": max_frontier,
                    "runtime_ms": (time.perf_counter() - start_time) * 1000,
                }

            r, c = current
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if grid[nr][nc]:
                    continue

                neighbor = (nr, nc)
                if neighbor in closed:
                    continue

                tentative_g = current_g + 1
                if tentative_g < g_score.get(neighbor, inf):
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    insertion_counter += 1
                    
                    f = tentative_g + self.heuristic_func(neighbor, goal)
                    heappush(open_heap, (f, -tentative_g, insertion_counter, neighbor))

            max_frontier = max(max_frontier, len(open_heap))

        return {
            "found": False,
            "path": None,
            "path_cost": inf,
            "nodes_expanded": nodes_expanded,
            "max_frontier": max_frontier,
            "runtime_ms": (time.perf_counter() - start_time) * 1000,
        }


# ---------------------------------------------------------
# Graph Generation
# ---------------------------------------------------------
def generate_graphs(summary_df, output_dir: Path):
    """Generate comparison bar charts as requested."""
    
    # 1. comparison_nodes_expanded.png
    plt.figure(figsize=(10, 6))
    bars = plt.bar(summary_df['heuristic'], summary_df['avg_nodes_expanded'], color='skyblue', edgecolor='black')
    plt.title('Average Nodes Expanded per Heuristic', fontsize=14, fontweight='bold')
    plt.xlabel('Heuristic', fontsize=12)
    plt.ylabel('Nodes Expanded', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.01), f'{yval:.1f}', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_nodes_expanded.png", dpi=150)
    plt.close()

    # 2. comparison_runtime.png
    plt.figure(figsize=(10, 6))
    bars = plt.bar(summary_df['heuristic'], summary_df['avg_runtime_ms'], color='lightgreen', edgecolor='black')
    plt.title('Average Runtime (ms) per Heuristic', fontsize=14, fontweight='bold')
    plt.xlabel('Heuristic', fontsize=12)
    plt.ylabel('Runtime (ms)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.01), f'{yval:.3f}', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_runtime.png", dpi=150)
    plt.close()
    
    # 3. metrics_summary.png (Combining Max Frontier and Path Cost in a multi-bar chart)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = range(len(summary_df['heuristic']))
    width = 0.35

    ax1.bar([i - width/2 for i in x], summary_df['avg_max_frontier'], width, label='Max Frontier Size', color='salmon', edgecolor='black')
    ax1.set_xlabel('Heuristic', fontsize=12)
    ax1.set_ylabel('Max Frontier Size', color='darkred', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='darkred')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df['heuristic'])

    ax2 = ax1.twinx()
    ax2.bar([i + width/2 for i in x], summary_df['avg_path_length'], width, label='Avg Path Length', color='gold', edgecolor='black')
    ax2.set_ylabel('Average Path Length', color='darkgoldenrod', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='darkgoldenrod')

    plt.title('Max Frontier and Path Cost Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png", dpi=150)
    plt.close()


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Customize default parameters
    ROWS = 25
    COLS = 25
    OBSTACLE_PROBABILITY = 0.20
    NUMBER_OF_TRIALS = 30
    RANDOM_SEED = 123
    MIN_START_GOAL_DISTANCE = 5
    
    OUTPUT_DIR = Path("astar_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    HEURISTICS = {
        "Dijkstra (h=0)": Heuristics.dijkstra,
        "Manhattan": Heuristics.manhattan,
        "Euclidean": Heuristics.euclidean,
        "Chebyshev": Heuristics.chebyshev,
        "Octile": Heuristics.octile
    }
    
    print_header("🤖 ROBOT PATH-FINDING USING A* SEARCH 🤖")

    print_card("Problem Configuration", [
        ("Grid Size", f"{ROWS} × {COLS} cells"),
        ("Obstacle Density", f"{int(OBSTACLE_PROBABILITY * 100)}%"),
        ("Benchmarking Trials", f"{NUMBER_OF_TRIALS}"),
        ("Min Start-Goal Distance", f"{MIN_START_GOAL_DISTANCE} cells"),
        ("Random Seed", f"{RANDOM_SEED}"),
        ("Output Folder", f"{OUTPUT_DIR}/"),
    ])

    print(f"\n{Style.YELLOW}⏳ Running benchmark experiments across heuristics...{Style.RESET}")

    # Run experiments
    records = []
    
    for trial in range(1, NUMBER_OF_TRIALS + 1):
        # Create a new GridWorld instance for each trial to keep random seeds consistent
        world = GridWorld(ROWS, COLS, OBSTACLE_PROBABILITY, seed=RANDOM_SEED + trial)
        world.generate_solvable_scenario(min_distance=MIN_START_GOAL_DISTANCE)
        
        for h_name, h_func in HEURISTICS.items():
            solver = AStarSolver(world, h_func)
            result = solver.solve()
            
            records.append({
                "trial": trial,
                "heuristic": h_name,
                "found": result["found"],
                "path_cost": result["path_cost"],
                "nodes_expanded": result["nodes_expanded"],
                "max_frontier": result["max_frontier"],
                "runtime_ms": result["runtime_ms"],
            })

    # Save and summarize
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "astar_results.csv", index=False)
    
    summary = df.groupby("heuristic").agg(
        success_rate=("found", "mean"),
        avg_path_length=("path_cost", "mean"),
        avg_nodes_expanded=("nodes_expanded", "mean"),
        avg_max_frontier=("max_frontier", "mean"),
        avg_runtime_ms=("runtime_ms", "mean"),
    ).reset_index()
    
    summary["success_rate"] *= 100
    summary.to_csv(OUTPUT_DIR / "astar_summary.csv", index=False)
    
    # Generate requested graphs
    print(f"{Style.YELLOW}📈 Generating performance comparison graphs...{Style.RESET}")
    generate_graphs(summary, OUTPUT_DIR)
    
    # Display Summary Table
    print_header("📊 BENCHMARK PERFORMANCE COMPARISON")

    headers = ["Heuristic", "Success", "Avg Cost", "Nodes Expanded", "Max Frontier", "Runtime (ms)"]
    row_fmt = "  │ {:<16} │ {:>8} │ {:>9} │ {:>15} │ {:>13} │ {:>12} │"
    div_line = "  ├─{:─<16}─┼─{:─<8}─┼─{:─<9}─┼─{:─<15}─┼─{:─<13}─┼─{:─<12}─┤".format("", "", "", "", "", "")
    top_line = "  ┌─{:─<16}─┬─{:─<8}─┬─{:─<9}─┬─{:─<15}─┬─{:─<13}─┬─{:─<12}─┐".format("", "", "", "", "", "")
    bot_line = "  └─{:─<16}─┴─{:─<8}─┴─{:─<9}─┴─{:─<15}─┴─{:─<13}─┴─{:─<12}─┘".format("", "", "", "", "", "")

    print(f"\n{top_line}")
    print(f"  │ {Style.BOLD}{headers[0]:<16}{Style.RESET} │ {Style.BOLD}{headers[1]:>8}{Style.RESET} │ {Style.BOLD}{headers[2]:>9}{Style.RESET} │ {Style.BOLD}{headers[3]:>15}{Style.RESET} │ {Style.BOLD}{headers[4]:>13}{Style.RESET} │ {Style.BOLD}{headers[5]:>12}{Style.RESET} │")
    print(div_line)

    for _, row in summary.iterrows():
        name = row["heuristic"]
        succ = f"{row['success_rate']:.1f}%"
        cost = f"{row['avg_path_length']:.2f}"
        nodes = f"{row['avg_nodes_expanded']:.1f}"
        front = f"{row['avg_max_frontier']:.1f}"
        runtime = f"{row['avg_runtime_ms']:.4f}"

        h_color = Style.GREEN if "Manhattan" in name else Style.WHITE
        print(f"  │ {h_color}{name:<16}{Style.RESET} │ {succ:>8} │ {cost:>9} │ {nodes:>15} │ {front:>13} │ {runtime:>12} │")

    print(bot_line)

    # Show example scenario
    print_header("🗺️  SAMPLE SOLVED PATH VISUALIZATION (Manhattan)")
    
    example_world = GridWorld(15, 25, OBSTACLE_PROBABILITY, seed=RANDOM_SEED + 999)
    example_world.generate_solvable_scenario(min_distance=10)
    
    solver = AStarSolver(example_world, Heuristics.manhattan)
    example_result = solver.solve()
    
    print(f"\n  {Style.BOLD}Legend:{Style.RESET}  {Style.GREEN}S{Style.RESET}=Start   {Style.RED}G{Style.RESET}=Goal   {Style.YELLOW}*{Style.RESET}=Optimal Path   {Style.DIM}#{Style.RESET}=Obstacle   {Style.DIM}·{Style.RESET}=Free Space\n")
    
    example_world.render(example_result["path"])
    
    print(f"\n  🎯 {Style.BOLD}Start Position :{Style.RESET} {Style.GREEN}{example_world.start}{Style.RESET}")
    print(f"  🏁 {Style.BOLD}Goal Position  :{Style.RESET} {Style.RED}{example_world.goal}{Style.RESET}")
    print(f"  📏 {Style.BOLD}Path Steps     :{Style.RESET} {Style.YELLOW}{example_result['path_cost']}{Style.RESET}")
    print(f"  ⚡ {Style.BOLD}Nodes Explored :{Style.RESET} {example_result['nodes_expanded']}")
    print(f"  ⏱️  {Style.BOLD}Search Time    :{Style.RESET} {example_result['runtime_ms']:.3f} ms\n")