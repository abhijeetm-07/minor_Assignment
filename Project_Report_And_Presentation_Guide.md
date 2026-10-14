# 🤖 Robot Pathfinding Using A* Search — Complete Project Report
**Artificial Intelligence Minor Project — Detailed Technical Documentation**

> **How to use this document**: This is your single, self-contained reference for understanding, presenting, and defending your project to a reviewer. Every concept, algorithm, data structure, heuristic, and code decision is explained here in detail. Read it top-to-bottom once, and you will be able to answer any question thrown at you.

---

## Table of Contents
1. [What Is This Project About?](#1-what-is-this-project-about)
2. [Background Concepts You Must Know](#2-background-concepts-you-must-know)
3. [File 1: Single Robot Pathfinding — Full Walkthrough](#3-file-1-single-robot-pathfinding)
4. [File 2: Multi-Robot Pathfinding — Full Walkthrough](#4-file-2-multi-robot-pathfinding)
5. [Heuristics — Deep Dive with Examples](#5-heuristics--deep-dive-with-examples)
6. [Key Code Explanations (Annotated Snippets)](#6-key-code-explanations)
7. [Graphs and Output Explained](#7-graphs-and-output-explained)
8. [Object-Oriented Architecture Explained](#8-object-oriented-architecture-explained)
9. [Frequently Asked Reviewer Questions (FAQ)](#9-frequently-asked-reviewer-questions)

---

## 1. What Is This Project About?

This project solves the **Robot Path Planning** problem using the **A\* (A-Star) Search Algorithm**. It is divided into two parts:

| Part | File | Goal |
|------|------|------|
| **Part 1** | `astar_robot_pathfinding.py` | Navigate **one robot** from Start → Goal on a 2D grid, avoiding obstacles. Compare 5 different heuristic functions to see which one finds the path fastest. |
| **Part 2** | `multi_robot_pathfinding.py` | Navigate **multiple robots** (3–5) simultaneously on the same grid, ensuring they **never collide** with each other. Compare a naïve baseline (Independent A\*) against a proposed method (CBA\* — Conflict-Based Adaptive A\*). |

**Why is this important in real life?**
- Warehouse robots (Amazon, Flipkart) must navigate without crashing.
- Self-driving cars must plan routes avoiding other vehicles.
- Video game characters need intelligent pathfinding around obstacles.

---

## 2. Background Concepts You Must Know

### 2.1 What Is a Grid World?

Imagine a chessboard. Each square (cell) is either:
- **Free** (the robot can walk on it), or
- **Blocked** (it's an obstacle; the robot cannot enter).

The robot starts at one cell (called **Start**) and must reach another cell (called **Goal**). It can move in **4 directions only**: Up, Down, Left, Right. Each step costs exactly **1 unit**.

A 5×5 grid example:
```
· · # · ·
· # · · ·
S · · # ·
· · # · ·
· · · · G
```
Here, `S` = Start, `G` = Goal, `#` = Obstacle, `·` = Free cell.

### 2.2 What Is a Search Algorithm?

A search algorithm systematically explores cells on the grid to find a path from Start to Goal. The key question is: **in what order do we explore cells?**

- **Breadth-First Search (BFS)**: Explore all cells at distance 1 from Start, then distance 2, then distance 3, etc. Like a ripple in water. Finds the shortest path but is very slow because it explores everything blindly.
- **Dijkstra's Algorithm**: Similar to BFS for uniform-cost grids. It always expands the cell with the cheapest known cost from the Start. Also finds the shortest path but explores too many cells.
- **A\* Search**: The "smart" version. It uses a **heuristic** — a mathematical guess of how far a cell is from the Goal — to prioritize exploring cells that *seem* closer to the Goal first. This dramatically reduces the number of cells explored.

### 2.3 The A\* Formula

At every step, A\* calculates a score for each cell:

$$f(n) = g(n) + h(n)$$

| Symbol | Meaning | Example |
|--------|---------|---------|
| $f(n)$ | **Total estimated cost** of the path through cell $n$. This is the score used to decide which cell to explore next. | If $g = 5$ and $h = 3$, then $f = 8$. |
| $g(n)$ | **Actual cost** from Start to cell $n$. This is exact — we know it because we've already walked this far. | If we took 5 steps to reach cell $n$, then $g(n) = 5$. |
| $h(n)$ | **Heuristic estimate** of the cost from cell $n$ to the Goal. This is a *guess*. | Manhattan distance from $n$ to Goal. |

**The Priority Queue (Min-Heap)**: A\* maintains a list of cells to explore, sorted by their $f(n)$ score. It always picks the cell with the **lowest** $f(n)$ first. This data structure is called a Min-Heap or Priority Queue.

### 2.4 What Does "Admissible" Mean?

A heuristic is **admissible** if it **never overestimates** the true shortest distance to the Goal.

- **Why it matters**: If the heuristic overestimates, A\* might skip over the actual shortest path, giving a wrong answer. An admissible heuristic guarantees A\* always finds the optimal (shortest) path.
- **Example**: If the true shortest path from cell $n$ to Goal is 7 steps, an admissible heuristic must return a value ≤ 7 (e.g., 5, 6, or 7 — but never 8 or more).

### 2.5 What Is "Node Expansion"?

When A\* "expands" a node, it means:
1. It **removes** the cell with the lowest $f(n)$ from the priority queue.
2. It **examines** all of that cell's neighbors (up to 4 neighbors on a grid).
3. It **adds** any unvisited or improved neighbors back into the priority queue.

The total number of expanded nodes directly measures how much "work" A\* did. Fewer expansions = faster algorithm.

### 2.6 What Is "Frontier" (Open List)?

The **frontier** is the set of cells that A\* knows about but hasn't explored yet. They are sitting in the priority queue, waiting their turn. The **maximum frontier size** tells us how much memory the algorithm used at its peak.

---

## 3. File 1: Single Robot Pathfinding

### 3.1 Configuration
```
Grid Size:              25 × 25 cells (625 total cells)
Obstacle Density:       20% (≈125 cells are obstacles)
Number of Trials:       30 independent experiments
Minimum Start-Goal Dist: 5 cells apart (ensures non-trivial paths)
Random Seed:            123 (for reproducibility)
Heuristics Tested:      5 (Dijkstra, Manhattan, Euclidean, Chebyshev, Octile)
```

### 3.2 Step-by-Step Execution Flow

**Step 1 — Grid Generation (`GridWorld.generate_grid`)**
- Create a 25×25 matrix.
- For every cell, generate a random number between 0 and 1. If the number < 0.20 (20%), mark it as an obstacle (`True`). Otherwise, mark it as free (`False`).
- Ensure at least 2 free cells exist (otherwise regenerate).

**Step 2 — Solvable Scenario Setup (`GridWorld.generate_solvable_scenario`)**
- Pick a random free cell as `Start`.
- Use **Breadth-First Search (BFS)** to find every free cell reachable from `Start`. This guarantees connectivity.
- From the reachable cells, filter only those that are at least 5 Manhattan distance away from `Start` (to avoid trivially short paths).
- Randomly pick one of these as `Goal`.
- **Key insight**: Because Goal is chosen from the BFS-reachable set, we are 100% guaranteed that a path exists. No trial is wasted on unsolvable grids.

**Step 3 — A\* Search (`AStarSolver.solve`)**
For each of the 5 heuristics, the same grid/start/goal is solved:

1. **Initialize**:
   - Put `Start` into the priority queue with $f = h(\text{Start}, \text{Goal})$ and $g = 0$.
   - Create a `g_score` dictionary: `{Start: 0}`.
   - Create a `parent` dictionary: `{Start: None}` (used to trace back the path).
   - Create an empty `closed` set (cells we've already fully processed).

2. **Main Loop** (repeat until goal is found or queue is empty):
   - **Pop** the cell with the lowest $f$ from the priority queue.
   - If this cell was already processed (`in closed`), skip it.
   - Mark it as processed (`closed.add(current)`). Increment `nodes_expanded`.
   - **Goal check**: If `current == goal`, reconstruct the path by following `parent` pointers backward from Goal to Start. Return success.
   - **Expand neighbors**: For each of the 4 directions (Up/Down/Left/Right):
     - Check if the neighbor is within grid bounds.
     - Check if the neighbor is not an obstacle.
     - Check if the neighbor hasn't already been processed.
     - Calculate `tentative_g = current_g + 1` (one step further).
     - If this new `g` is better than any previously known `g` for this neighbor, update `g_score[neighbor]`, set `parent[neighbor] = current`, calculate `f = tentative_g + h(neighbor, goal)`, and push to the priority queue.
   - Track the maximum size of the priority queue (`max_frontier`).

3. **Path Reconstruction** (`_reconstruct_path`):
   - Start at `Goal`.
   - Follow `parent[Goal]` → `parent[parent[Goal]]` → ... → `Start`.
   - Reverse the list to get Start → Goal order.

**Step 4 — Data Collection**
After all 30 trials × 5 heuristics = 150 runs, the results are stored in a Pandas DataFrame with columns: `trial`, `heuristic`, `found`, `path_cost`, `nodes_expanded`, `max_frontier`, `runtime_ms`.

**Step 5 — Summary & Graphs**
- Group by heuristic, compute averages.
- Save CSVs: `astar_results.csv`, `astar_summary.csv`.
- Generate 3 bar charts using matplotlib: `comparison_nodes_expanded.png`, `comparison_runtime.png`, `metrics_summary.png`.

**Step 6 — Terminal Visualization**
- A separate 15×25 grid is generated and solved using Manhattan.
- The grid is printed to the terminal using ANSI color codes: `S` (green) = Start, `G` (red) = Goal, `*` (yellow) = optimal path cells, `#` (dim) = obstacles, `·` (dim) = free cells.

---

## 4. File 2: Multi-Robot Pathfinding

### 4.1 The Problem Statement

When you have multiple robots on the same grid, **independent planning doesn't work**. Even if each robot finds its optimal path, they may crash into each other because they don't know about each other.

Two types of collisions:
1. **Vertex Collision**: Two robots occupy the **same cell at the same time step**.
   - Example: Robot A is at (5, 3) at time T=4. Robot B is also at (5, 3) at time T=4. → **Crash!**
2. **Edge-Swap Collision**: Two robots pass **through each other** on the same edge.
   - Example: At time T=4, Robot A is at (5, 3) and Robot B is at (5, 4). At time T=5, Robot A moves to (5, 4) and Robot B moves to (5, 3). They swap through each other. → **Crash!**

### 4.2 Configuration
```
Grid Sizes:             15×15 and 20×20
Obstacle Densities:     15% and 25%
Number of Agents:       3 and 5
Total Configurations:   2 × 2 × 2 = 8
Trials per Config:      5
Max High-Level Expansions: 200 (safety limit for CBA*)
Min Start-Goal Distance:   6 cells
```

### 4.3 Method 1: Independent A\* (Baseline)

This is the **naïve approach**. It is simple but flawed.

**How it works:**
1. For each robot, run standard A\* independently (as if it's the only robot on the grid).
2. Collect all paths.
3. After all paths are computed, scan for collisions by checking every time step.
4. If any collision is found, mark the run as a **failure**.

**Why it fails:**
- Robots have no awareness of each other during planning.
- On a crowded grid with 5 agents, there is roughly a **45% failure rate** due to inevitable collisions.
- It is fast (low planning time) but unreliable.

### 4.4 Method 2: CBA\* — Conflict-Based Adaptive A\* (Proposed)

This is the **intelligent approach**. It guarantees collision-free paths.

CBA\* is a **two-level hierarchical algorithm**:

#### Level 1: High-Level Search (Conflict Tree)

Think of this as a "project manager" that detects conflicts and assigns constraints.

**Data Structures:**
- A **Constraint Tree** (CT) node contains:
  - `paths`: the current planned paths for all robots.
  - `constraints`: a list of forbidden (time, cell) entries for each robot.
  - `cost`: the total sum of all robots' path lengths.
  - `conflicts`: the number of remaining collision events.

**Algorithm:**
1. **Root Node**: Plan all robots independently using A\*. These paths might have collisions.
2. **Detect the Most Severe Conflict**:
   - Scan all paths across all time steps.
   - For every pair of robots, check for vertex and edge-swap collisions.
   - **Adaptive Severity Scoring** (this is what makes CBA\* unique):
     - Vertex collisions are scored higher (10.0 base) than edge swaps (6.0 base).
     - Collisions in areas with **high local obstacle density** are scored even higher (+4.0 × density). Why? Because in tight corridors with lots of obstacles, the conflict is harder to resolve, so we should address it first.
     - Earlier-time conflicts are slightly deprioritized (+0.01 × remaining time). Why? Because resolving later-time conflicts first tends to cascade fewer changes.
   - Pick the conflict with the **highest severity score**.
3. **Branch**: For the chosen conflict between Robot $i$ and Robot $j$:
   - **Branch A**: Add a constraint to Robot $i$: "You are forbidden from entering cell $(X, Y)$ at time $T$." Then **replan only Robot $i$** using the Low-Level Search.
   - **Branch B**: Add a constraint to Robot $j$ instead. Replan only Robot $j$.
   - Both branches are pushed to the high-level priority queue.
4. **Priority**: The queue is sorted by `cost + 10 × remaining_conflicts`. This ensures we explore solutions that are both cheap AND conflict-free first.
5. **Termination**: When a CT node has **zero conflicts**, we have found a valid, collision-free solution!

#### Level 2: Low-Level Search (Space-Time A\*)

This is a modified A\* that operates in **3 dimensions**: `(row, col, time)`.

**Key Differences from Standard A\*:**
- **State**: Instead of just `(row, col)`, the state is `(row, col, time)`. This means being at cell (5, 3) at time T=4 is a **different state** from being at (5, 3) at time T=5.
- **Wait Action**: The robot has **5 possible moves** instead of 4:
  - Up `(-1, 0)`, Down `(1, 0)`, Left `(0, -1)`, Right `(0, 1)`, **Wait `(0, 0)`**.
  - The Wait action means the robot stays in its current cell but time advances by 1. This lets other robots pass by safely.
- **Constraint Checking**: Before expanding to a neighbor, the algorithm checks:
  - Is `(next_time, neighbor_cell)` in the vertex constraint set? If yes, skip this neighbor.
  - Is `(current_time, current_cell, neighbor_cell)` in the edge constraint set? If yes, skip.
- **Goal Condition**: The robot doesn't just need to reach the goal cell — it also needs to ensure there are no future constraints that would force it off the goal after arrival.
- **Horizon**: The search is capped at a maximum time horizon to prevent infinite searching.

**Walkthrough Example:**
Suppose Robot 1 originally passes through cell (5, 3) at time T=4. But the High-Level Search says: "Robot 1, you are FORBIDDEN from (5, 3) at T=4."

Now, Low-Level A\* replans Robot 1:
- At time T=3, Robot 1 is at (5, 2), about to move right to (5, 3).
- The algorithm sees the constraint: "Cannot enter (5, 3) at T=4."
- Instead of moving right, it chooses **Wait**: stay at (5, 2) at T=4.
- At T=5, the other robot has passed, so Robot 1 can now safely move to (5, 3).
- The path is 1 step longer (due to the wait), but it is collision-free.

### 4.5 Collision Detection (`detect_collisions`)

After paths are computed, this function scans every time step from T=0 to T=makespan:

**Vertex Collision Check:**
```
For each time step T:
    Create a dictionary: {cell → [list of robot IDs at that cell]}
    If any cell has more than one robot → COLLISION!
```

**Edge-Swap Collision Check:**
```
For each time step T:
    For every pair of robots (i, j):
        If Robot i moves from A→B AND Robot j moves from B→A:
            → They swap through each other. COLLISION!
```

### 4.6 Benchmark Results Interpretation

The output table shows:

| Metric | Independent A\* | CBA\* (Proposed) |
|--------|----------------|------------------|
| **Success Rate** | ~55% | **100%** |
| **Avg Collisions** | ~0.68 per run | **0.00** |
| **Avg Makespan** | `inf` (fails often) | ~20 time steps |
| **Planning Time** | ~0.2 ms (fast but wrong) | ~0.8 ms (slower but correct) |

**Key takeaway for reviewer**: CBA\* trades a small amount of extra planning time for a 100% collision-free guarantee. In real-world robotics, correctness is far more important than speed.

---

## 5. Heuristics — Deep Dive with Examples

Consider a robot at cell **(2, 3)** trying to reach goal **(7, 6)**.
- $dx = |2 - 7| = 5$
- $dy = |3 - 6| = 3$

### 5.1 Dijkstra ($h = 0$)
- **Formula**: $h(n) = 0$
- **Value**: 0
- **What it does**: Gives NO directional guidance. A\* behaves like Dijkstra's algorithm, exploring cells in concentric circles radiating outward from Start. It explores **every cell at every distance** before finding the goal.
- **Admissible?** Yes (0 never overestimates anything).
- **Performance**: Finds the optimal path but expands the **most nodes** (≈265 average in our tests). Very slow.
- **When to use**: When you want guaranteed optimality and don't care about speed. Also serves as a benchmark — every other heuristic should expand fewer nodes than this.

### 5.2 Manhattan Distance
- **Formula**: $h(n) = |x_1 - x_2| + |y_1 - y_2|$
- **Value**: $5 + 3 = 8$
- **What it does**: Counts the minimum number of horizontal + vertical steps needed, assuming NO obstacles. It is the exact cost of a staircase-shaped path on a clear grid.
- **Admissible?** Yes. On a 4-directional grid, you can never reach the goal in fewer than Manhattan steps. Obstacles can only make the path longer, never shorter.
- **Performance**: Expands the **fewest nodes** (≈27 average). This is because on a 4-directional grid, Manhattan is the **tightest possible admissible heuristic** — its estimate exactly equals the true shortest-path cost on an obstacle-free grid. The closer $h(n)$ is to the true cost, the fewer nodes A\* needs to expand.
- **When to use**: **Always**, for 4-directional grid movement. It is the gold standard.

### 5.3 Euclidean Distance
- **Formula**: $h(n) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$
- **Value**: $\sqrt{25 + 9} = \sqrt{34} ≈ 5.83$
- **What it does**: Calculates the straight-line distance, as if the robot could fly diagonally at any angle.
- **Admissible?** Yes. The straight line is always shorter than or equal to any path on a grid (which must follow grid lines).
- **Performance**: Expands ≈73 nodes on average. It **underestimates** compared to Manhattan because it assumes diagonal shortcuts that don't exist on a 4-way grid. This means A\* is less "focused" and explores more cells in wrong directions before finding the goal.
- **Why is Euclidean < Manhattan?** Manhattan says "you need 8 steps." Euclidean says "you only need 5.83 steps." The true answer is 8 (on an obstacle-free grid). So Euclidean is a weaker (less informative) heuristic.

### 5.4 Chebyshev Distance
- **Formula**: $h(n) = \max(|x_1 - x_2|, |y_1 - y_2|)$
- **Value**: $\max(5, 3) = 5$
- **What it does**: Also called "chessboard distance" because it's how a King moves in chess — it can move diagonally, and each diagonal step costs 1. It counts the minimum number of "King moves" to reach the goal.
- **Admissible?** Yes. Since diagonal moves are not allowed on our 4-way grid, Chebyshev always underestimates (5 < 8).
- **Performance**: Expands ≈83 nodes on average. Even more of an underestimate than Euclidean, so A\* explores even more wrong directions.
- **Why include it?** It demonstrates that heuristics designed for 8-directional movement perform worse on 4-directional grids.

### 5.5 Octile Distance
- **Formula**: $h(n) = \max(dx, dy) + (\sqrt{2} - 1) \times \min(dx, dy)$
- **Value**: $\max(5, 3) + (1.414 - 1) \times \min(5, 3) = 5 + 0.414 \times 3 = 5 + 1.242 = 6.24$
- **What it does**: This is the optimal heuristic for 8-directional movement where diagonal steps cost $\sqrt{2} ≈ 1.414$ and cardinal steps cost 1. It's a hybrid between Manhattan and Chebyshev. It takes as many diagonal steps as possible (which cover both dx and dy simultaneously) and then finishes with cardinal steps.
- **Admissible?** Yes. On a 4-way grid (where diagonal movement isn't allowed), the true cost is at least Manhattan (8), and Octile (6.24) is strictly less.
- **Performance**: Expands ≈67 nodes. Better than Chebyshev and Euclidean because it's a tighter estimate, but still worse than Manhattan.
- **Why include it?** It demonstrates a "4th unique admissible heuristic" as requested by the assignment. It's commonly used in video games and robotics where diagonal movement IS allowed.

### 5.6 Heuristic Comparison Summary

| Heuristic | Formula | Value (Example) | Nodes Expanded (Avg) | Runtime (ms) | Tightness |
|-----------|---------|----------------|---------------------|-------------|-----------|
| Dijkstra | 0 | 0 | 265.7 (worst) | 0.315 (worst) | None |
| Chebyshev | max(dx, dy) | 5 | 82.9 | 0.114 | Weak |
| Euclidean | sqrt(dx²+dy²) | 5.83 | 72.7 | 0.105 | Moderate |
| Octile | max + (√2-1)×min | 6.24 | 66.7 | 0.104 | Good |
| **Manhattan** | **dx + dy** | **8** | **26.6 (best)** | **0.045 (best)** | **Perfect** |

**Key insight for reviewer**: The closer the heuristic value is to the true cost, the fewer nodes A\* expands. Manhattan = 8 matches the true cost perfectly on a 4-way grid, making it the optimal choice.

---

## 6. Key Code Explanations

### 6.1 The A\* Core Loop (Single Robot)

```python
while open_heap:                                  # While there are cells to explore
    f_score, negative_g, _, current = heappop(open_heap)  # Pop lowest f(n)
    
    if current in closed:   # Already fully processed? Skip.
        continue
    
    closed.add(current)     # Mark as processed
    nodes_expanded += 1     # Count the work done
    
    if current == goal:     # FOUND IT!
        path = self._reconstruct_path(parent, goal)
        return {"found": True, "path": path, ...}
    
    # Explore all 4 neighbors
    r, c = current
    for dr, dc in moves:    # moves = [(-1,0), (1,0), (0,-1), (0,1)]
        nr, nc = r + dr, c + dc
        
        if not (0 <= nr < rows and 0 <= nc < cols):  # Out of bounds?
            continue
        if grid[nr][nc]:     # Is it an obstacle?
            continue
        
        neighbor = (nr, nc)
        if neighbor in closed:  # Already processed?
            continue
        
        tentative_g = current_g + 1  # One step further from Start
        
        if tentative_g < g_score.get(neighbor, inf):  # Is this a better path?
            g_score[neighbor] = tentative_g       # Update best known cost
            parent[neighbor] = current            # Remember how we got here
            f = tentative_g + heuristic(neighbor, goal)  # f = g + h
            heappush(open_heap, (f, -tentative_g, insertion_counter, neighbor))
```

**Why `-tentative_g` in the heap?**
When two cells have the same $f$ score (a tie), we want to prefer the cell that is **deeper** (has a larger $g$). Using `-tentative_g` ensures that deeper cells are popped first during ties. This is a standard tie-breaking technique that makes A\* slightly more efficient.

**Why `insertion_counter`?**
Python's `heapq` can crash if two heap entries are equal and the items inside are not comparable (tuples of cells). The insertion counter is a unique integer that breaks any remaining ties, ensuring the heap always has a valid ordering.

### 6.2 Path Reconstruction

```python
def _reconstruct_path(self, parent, current):
    path = []
    while current is not None:
        path.append(current)
        current = parent.get(current)  # Follow the chain backward
    path.reverse()  # Reverse: Goal→Start becomes Start→Goal
    return path
```

**How `parent` works**: During the search, whenever we find a better path to a cell, we record `parent[cell] = where_we_came_from`. After reaching the Goal, we follow this chain backward to trace the exact path.

### 6.3 BFS for Reachability Check

```python
def get_reachable_cells(self, start):
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
```

**Why BFS and not A\*?** We don't need a shortest path here — we just need to find ALL cells that are reachable from Start. BFS is perfect for this: it explores outward layer by layer, visiting every connected free cell exactly once. The result is used to ensure the Goal is always reachable.

### 6.4 CBA\* High-Level Branching (Multi-Robot)

```python
# A conflict was detected between Robot i and Robot j at cell (X,Y) at time T.
for constrained_agent in (i, j):  # Try constraining each robot in turn
    
    # Copy all constraints and add a new one
    child_constraints = [c.copy() for c in constraints]
    
    if conflict_type == "vertex":
        # "Robot X, you are FORBIDDEN from cell (X,Y) at time T"
        child_constraints[constrained_agent].vertex.add((conflict_time, cell))
    else:
        # "Robot X, you are FORBIDDEN from traversing edge A→B at time T"
        child_constraints[constrained_agent].edge.add((conflict_time, edge_from, edge_to))
    
    # Replan ONLY the constrained robot using Low-Level Space-Time A*
    new_path, expanded = self.low_level_constrained_astar(
        self.world.agents[constrained_agent].start,
        self.world.agents[constrained_agent].goal,
        child_constraints[constrained_agent],
        horizon
    )
    
    # If replanning succeeded and path is valid, add to priority queue
    child_paths = list(paths)
    child_paths[constrained_agent] = new_path
    child_cost = sum(len(path) - 1 for path in child_paths)
    child_conflicts = self.detect_collisions(child_paths)["collision_events"]
    
    # Priority = cost + penalty for remaining conflicts
    heappush(open_heap, (child_cost + 10.0 * child_conflicts, ...))
```

### 6.5 Low-Level Space-Time A\* (Multi-Robot)

```python
# The state is 3D: (row, col, time)
for dr, dc in SPACE_TIME_MOVES:  # Includes WAIT (0, 0)!
    nr, nc = r + dr, c + dc
    next_time = t + 1  # Time always advances
    
    # Grid boundary and obstacle checks (same as normal A*)
    if not (0 <= nr < self.world.rows and 0 <= nc < self.world.cols):
        continue
    if self.world.grid[nr][nc]:
        continue
    
    nxt = (nr, nc)
    
    # NEW: Check constraints from High-Level Search
    # "Am I forbidden from entering 'nxt' at 'next_time'?"
    if not self._state_respects_constraints(constraints, current, nxt, next_time):
        continue  # This move is BLOCKED by a constraint
    
    next_state = (nr, nc, next_time)  # 3D state!
    tentative_g = current_g + 1
    
    if tentative_g < g_score.get(next_state, inf):
        g_score[next_state] = tentative_g
        parent[next_state] = state
        h = Heuristics.manhattan(nxt, goal)
        heappush(open_heap, (tentative_g + h, -tentative_g, counter, next_state))
```

**Why is the Wait action crucial?**
Without Wait, if a robot is blocked by a constraint, it must find a completely different route around the obstacle (which may not exist in tight corridors). With Wait, the robot can simply pause for 1 time step, let the other robot pass, and then continue on its original optimal route. This dramatically increases the success rate of conflict resolution.

---

## 7. Graphs and Output Explained

### 7.1 Single Robot Graphs (in `astar_results/`)

| File | What It Shows |
|------|--------------|
| `comparison_nodes_expanded.png` | Bar chart comparing the average number of nodes expanded by each heuristic. Manhattan should be the shortest bar (fewest nodes). Dijkstra should be the tallest (most nodes). |
| `comparison_runtime.png` | Bar chart comparing average runtime (ms). Mirrors the nodes_expanded chart — more nodes = more time. |
| `metrics_summary.png` | Dual-axis bar chart. Left axis: Max Frontier Size. Right axis: Average Path Length. Path length should be identical for all heuristics (they all find optimal paths), but frontier size varies. |

### 7.2 Multi-Robot Graphs (in `multi_robot_results/`)

| File | What It Shows |
|------|--------------|
| `success_rate_vs_agents.png` | Line chart. As agent count increases, Independent A\* success drops. CBA\* stays at 100%. |
| `collisions_vs_agents.png` | Line chart. Independent A\* collisions increase with more agents. CBA\* remains at 0. |
| `makespan_vs_agents.png` | Line chart. Average makespan (longest individual path time) for successful runs. |
| `planning_time_vs_agents.png` | Line chart. CBA\* takes slightly more time than Independent A\* due to the conflict resolution overhead. |
| `success_rate_vs_obstacles.png` | Line chart grouped by obstacle density (15% vs 25%). Higher density → harder problem. |
| `planning_time_vs_grid.png` | Line chart grouped by grid size (15×15 vs 20×20). Larger grids → more planning time. |

### 7.3 CSV Files

| File | Contents |
|------|----------|
| `astar_results.csv` | Raw per-trial data for all 150 runs (30 trials × 5 heuristics). |
| `astar_summary.csv` | Aggregated averages per heuristic. |
| `multi_robot_results.csv` | Raw per-trial data for all configurations. |
| `multi_robot_summary.csv` | Aggregated averages per (grid_size, obstacle%, agents, method). |
| `multi_robot_overall_comparison.csv` | Grand averages for Independent A\* vs CBA\*. |

---

## 8. Object-Oriented Architecture Explained

### 8.1 Why OOP?

The original code was procedural — all functions were loose, global, and intertwined. The refactored code uses classes to group related functionality, making it:
- **Easier to read**: Everything about the grid is in `GridWorld`, everything about solving is in `AStarSolver`.
- **Easier to test**: You can create a `GridWorld` object and test it independently.
- **Easier to extend**: Adding a new heuristic means just adding one static method to the `Heuristics` class.

### 8.2 Class Responsibilities

**File 1 (`astar_robot_pathfinding.py`):**

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `GridWorld` | Manages the 2D grid, generates obstacles, validates reachability, renders the terminal visualization. | `generate_grid()`, `get_reachable_cells()`, `generate_solvable_scenario()`, `render()` |
| `AStarSolver` | Takes a `GridWorld` and a heuristic function. Runs A\* and returns metrics. | `solve()`, `_reconstruct_path()` |
| `Heuristics` | Contains 5 static methods, one per heuristic function. | `dijkstra()`, `manhattan()`, `euclidean()`, `chebyshev()`, `octile()` |

**File 2 (`multi_robot_pathfinding.py`):**

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `GridWorld` | Manages the grid AND multi-agent scenario generation. Also handles local obstacle density calculation for CBA\*'s adaptive conflict scoring. | `generate_scenario()`, `local_obstacle_density()`, `render()` |
| `MultiAgentSolver` | Contains all solving logic: single-agent A\*, collision detection, Independent A\*, and the full CBA\* algorithm. | `astar_single()`, `detect_collisions()`, `independent_astar()`, `cba_star()`, `low_level_constrained_astar()`, `find_adaptive_conflict()` |
| `ExperimentRunner` | Orchestrates batch experiments across configurations, computes analytics, and generates all plots. | `run_experiments()`, `summarize()`, `generate_plots()`, `save_example()` |
| `AgentScenario` | Data container for a robot's start and goal positions. | (dataclass) |
| `AgentConstraints` | Stores space-time vertex and edge constraints for one robot during CBA\*. | `copy()` |

---

## 9. Frequently Asked Reviewer Questions

### Q1: "Why did you choose A\* over other algorithms like BFS or Dijkstra?"
**A**: BFS and Dijkstra find optimal paths but explore cells blindly in all directions. A\* uses a heuristic to guide the search toward the goal, dramatically reducing the number of cells explored. Our benchmarks show Manhattan-guided A\* expands only 27 nodes vs. Dijkstra's 266 — a 10× improvement.

### Q2: "Are all your heuristics admissible? How do you know?"
**A**: Yes. On a 4-directional grid where each step costs 1:
- **Dijkstra (0)**: Obviously ≤ any real cost.
- **Manhattan**: Exactly equals the obstacle-free shortest path. Obstacles can only increase the real cost.
- **Euclidean**: Straight-line distance ≤ any grid path (triangle inequality).
- **Chebyshev**: Assumes free diagonal movement (cost 1), which our grid doesn't allow, so it always underestimates.
- **Octile**: Assumes diagonal costs √2, still less than the 2-step Manhattan cost of a diagonal on a 4-way grid.

### Q3: "Why does Manhattan perform the best?"
**A**: Because on a 4-directional grid, Manhattan distance is the **tightest** admissible heuristic. The tighter the heuristic (closer to the true cost), the fewer nodes A\* needs to explore. Manhattan's estimate exactly matches the true cost on an obstacle-free grid, leaving almost no room for A\* to explore unnecessary cells.

### Q4: "What is novel about CBA\* compared to standard CBS?"
**A**: Three things:
1. **Adaptive conflict severity scoring**: Instead of resolving the first conflict found, CBA\* resolves the most *severe* conflict first, considering local obstacle density and conflict persistence.
2. **Selective replanning**: Only the constrained robot is replanned, not all robots.
3. **Conflict-aware priority ordering**: The high-level queue sorts by `cost + 10 × conflicts`, preferring solutions that are both cheap and conflict-free.

### Q5: "What happens if CBA\* can't find a solution?"
**A**: CBA\* has a safety limit of 200 high-level expansions. If it exhausts this limit without finding a collision-free solution, it returns a failure. This prevents the algorithm from running forever on extremely difficult instances. In our benchmarks, CBA\* achieved 100% success rate within this limit.

### Q6: "Why do you use the `heapq` module? What is a heap?"
**A**: A heap (specifically a min-heap) is a data structure that always gives you the smallest element in O(log n) time. A\* needs to constantly pick the cell with the lowest $f(n)$ — a heap does this efficiently. Without it, we'd have to scan the entire list every time (O(n)), which is much slower.

### Q7: "What is the time complexity of A\*?"
**A**: In the worst case, A\* can expand every cell in the grid: O(V log V), where V is the number of cells. With a good heuristic like Manhattan, the practical complexity is much lower because most cells are never expanded. The `log V` factor comes from heap operations.

### Q8: "Why does the Independent A\* success rate show 'inf' for makespan?"
**A**: When Independent A\* produces paths with collisions, we mark the run as a *failure*. Failed runs have no valid makespan, so we record it as infinity (`inf`). The average of values containing infinity is infinity. This clearly communicates that the baseline method cannot reliably produce usable paths.

---

*End of Document. You are now fully prepared to present and defend this project.*
