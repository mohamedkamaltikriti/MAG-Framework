# ====================================================================
# RUN FILE: run_uy734.py
# --------------------------------------------------------------------
# Documentation of the Adaptive Advanced Magnetic Algorithm (Abstract MAG)
# on the large Traveling Salesman Problem (TSP) instance: uy734 (734 cities).
# 
# Test Objectives:
# 1. To prove the Scalability of the Abstract MAG algorithm.
# 2. To determine the optimal Local Search heuristic (2-Opt vs. 3-Opt)
#    for the algorithm's implementation in a Python environment.
# 
# Best Single Run Result (60.0 seconds) achieved with 2-Opt:
# - Final Cost: 93757.00
# - Gap from Optimal (79114): 18.51%
# - Industry Benchmark (Google OR-Tools): 82636.00 (4.45% Gap)
# 
# Conclusion: 2-Opt proved superior to the computationally expensive 3-Opt
#             for achieving a higher number of effective search iterations.
# ====================================================================

import numpy as np
import math
import time
import random

# ====================================================================
# 1. TSPLIB Data Loading Function (NINT Approximation)
# ====================================================================
def load_tsplib_data(file_path):
    """Reads TSPLIB data from a file and returns the distance matrix."""
    coordinates = []
    reading_coords = False
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    for line in lines:
        line = line.strip()
        if line == 'NODE_COORD_SECTION':
            reading_coords = True
            continue
        if reading_coords and line.startswith('EOF'):
            break
        if reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates.append((x, y))
                except ValueError:
                    continue
    
    N_CITIES = len(coordinates)
    dist = np.zeros((N_CITIES, N_CITIES))
    
    # Calculate Euclidean distance using NINT approximation
    for i in range(N_CITIES):
        for j in range(i + 1, N_CITIES):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            dist[i, j] = dist[j, i] = int(distance + 0.5) 
            
    return dist

# ====================================================================
# 2. Core Helper Functions (Cost, Greedy, Kick)
# ====================================================================
def tsp_cost(path, dist):
    n = len(path)
    return sum(dist[path[i], path[(i + 1) % n]] for i in range(n))

def greedy_tsp(dist):
    n = len(dist)
    path = [0]
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    for _ in range(1, n):
        last = path[-1]
        unvisited = np.where(~visited)[0]
        if not unvisited.size: break
        distances_to_unvisited = dist[last, unvisited]
        next_city_index_in_unvisited = np.argmin(distances_to_unvisited)
        next_city = unvisited[next_city_index_in_unvisited]
        path.append(next_city)
        visited[next_city] = True
    return np.array(path)

def tsp_kick(best_path, dist):
    """Performs a random perturbation (kick) to escape local optima."""
    n = len(best_path)
    i, j, k, l = sorted(random.sample(range(n), 4))
    new_path = np.concatenate([best_path[:i], best_path[j:k+1], best_path[l:], best_path[k+1:l], best_path[i:j]])
    return new_path[:n]

# ====================================================================
# 3. Optimized Local Search Function (2-Opt)
# ====================================================================
def tsp_neighbor_2opt(path, dist, alpha):
    """Performs a randomized 2-Opt local search."""
    n = len(path)
    current_cost = tsp_cost(path, dist)
    best_path = path.copy()
    improved = False
    
    # Number of probes determined by alpha parameter
    probes = n * alpha 
    
    for _ in range(int(probes)):
        # Select two random cut points
        i, j = sorted(random.sample(range(n), 2))
        if j - i < 2:
            continue
            
        # Create candidate path by reversing the segment between i and j
        candidate = np.concatenate((path[:i], path[i:j+1][::-1], path[j+1:]))
        cost = tsp_cost(candidate, dist)
        
        # Accept improvement
        if cost < current_cost:
            current_cost = cost
            best_path = candidate
            improved = True
            
    return best_path, current_cost, improved

# ====================================================================
# 4. Abstract MAG Class (using 2-Opt)
# ====================================================================
class AbstractMAG:
    def __init__(self, problem_data, alpha=2, kick_every=15, time_limit=5.0):
        self.obj_func = tsp_cost 
        self.neighbor = tsp_neighbor_2opt # Using 2-Opt for best performance
        self.kick = tsp_kick 
        self.data = problem_data
        self.alpha = alpha
        self.kick_every = kick_every
        self.time_limit = time_limit

    def solve(self, initial_solution):
        path = initial_solution.copy()
        best_path = path.copy()
        best_cost = self.obj_func(path, self.data)

        start = time.time()
        kick_count = 0

        while time.time() - start < self.time_limit:
            improved = True
            while improved and time.time() - start < self.time_limit:
                path, cost, improved = self.neighbor(path, self.data, self.alpha)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path.copy()

            kick_count += 1
            if kick_count >= self.kick_every:
                path = self.kick(best_path, self.data)
                kick_count = 0

        return best_path, best_cost


# ====================================================================
# 5. Main Execution and Benchmarking
# ====================================================================
if __name__ == "__main__":
    
    # Configuration
    FILE_PATH = "/content/drive/MyDrive/uy734.tsp"  
    TIME_TO_RUN = 60.0 
    BEST_KNOWN_COST = 79114 

    print(f"Loading data from {FILE_PATH}...")
    dist_matrix = load_tsplib_data(FILE_PATH)
    if dist_matrix is None:
        exit()

    N_CITIES = len(dist_matrix)
    print(f"✅ Data loaded: {N_CITIES} cities for Uy734 (TSPLIB).")

    print("Running Greedy for initial solution...")
    initial = greedy_tsp(dist_matrix)
    initial_cost = tsp_cost(initial, dist_matrix)
    print(f"Greedy cost: {initial_cost:.2f}")

    print(f"Running Abstract MAG with 2-Opt optimization (60.0 seconds)...")
    solver = AbstractMAG(dist_matrix, time_limit=TIME_TO_RUN) 
    best_path, best_cost = solver.solve(initial)

    print(f"\n✅ Abstract MAG finished execution on Uy734")
    print(f"Best Final Cost: {best_cost:.2f}")
    
    gap = ((best_cost - BEST_KNOWN_COST) / BEST_KNOWN_COST) * 100
    print(f"Gap from known optimal ({BEST_KNOWN_COST}): {gap:.2f}%")

