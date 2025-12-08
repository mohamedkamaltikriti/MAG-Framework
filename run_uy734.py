# ====================================================================
# 📄 التقرير النهائي للأداء لخوارزمية Abstract MAG
# --------------------------------------------------------------------
# 
# الابتكار العلمي: الخوارزمية المغناطيسية المتقدمة المتكيفة (Abstract MAG).
# مسألة الاختبار: uy734 (734 مدينة) - الحل الأمثل المعروف: 79114.
# 
# ⚙️ المعاملات المُثلى:
#    - معامل البحث العميق (Alpha - α): 100
#    - تكرار القفزة الاستكشافية (Kick Every): 5
# 
# 📊 مقارنة الأداء:
#    - مدة 60 ثانية: التكلفة 92010.00 | الفجوة 16.30%
#    - مدة 120 ثانية: التكلفة 90324.00 | الفجوة 14.17%
# 
# تحدي التنفيذ: تم استخدام 2-Opt Python بدلاً من 3-Opt C++ المُسرَّع 
#              بسبب فشل الربط التقني (Pybind11/CMake) في بيئة التشغيل.
# 
# --------------------------------------------------------------------
# 📄 FINAL PERFORMANCE REPORT for Abstract MAG Algorithm
# --------------------------------------------------------------------
# 
# Scientific Innovation: Adaptive Advanced Magnetic Algorithm (Abstract MAG).
# Problem Instance: uy734 (734 Cities) - Best Known Optimal Cost: 79114.
# 
# ⚙️ Optimal Parameters:
#    - Deep Search Factor (Alpha - α): 100
#    - Kick Frequency (Kick Every): 5
# 
# 📊 Performance Comparison:
#    - 60 seconds runtime: Final Cost 92010.00 | Gap 16.30%
#    - 120 seconds runtime: Final Cost 90324.00 | Gap 14.17%
# 
# Implementation Challenge: 2-Opt Python was used instead of the accelerated 
#                           3-Opt C++ due to a technical linking failure 
#                           (Pybind11/CMake error) in the execution environment.
# 
# ====================================================================

import numpy as np
import math
import time
import random

# ====================================================================
# 1. دالة تحميل بيانات TSPLIB / TSPLIB Data Loading Function
# ====================================================================
def load_tsplib_data(file_path):
    """Loads coordinates and calculates the distance matrix (NINT approximation)."""
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
    
    for i in range(N_CITIES):
        for j in range(i + 1, N_CITIES):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            dist[i, j] = dist[j, i] = int(distance + 0.5) 
            
    return dist

# ====================================================================
# 2. الدوال المساعدة الأساسية / Core Helper Functions
# ====================================================================
def tsp_cost(path, dist):
    """Calculates the total cost (distance) of a given path."""
    n = len(path)
    return sum(dist[path[i], path[(i + 1) % n]] for i in range(n))

def greedy_tsp(dist):
    """Generates a greedy initial solution for TSP."""
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
    """Performs a random perturbation (Kick) to escape local optima (4-Opt Style)."""
    n = len(best_path)
    i, j, k, l = sorted(random.sample(range(n), 4))
    new_path = np.concatenate([best_path[:i], best_path[j:k+1], best_path[l:], best_path[k+1:l], best_path[i:j]])
    return new_path[:n]

# ====================================================================
# 3. دالة البحث الجواري المُحسَّنة / Optimized Local Search Function (2-Opt)
# ====================================================================
def tsp_neighbor_2opt(path, dist, alpha):
    """Performs a randomized 2-Opt local search using alpha probes."""
    n = len(path)
    current_cost = tsp_cost(path, dist)
    best_path = path.copy()
    improved = False
    
    probes = n * alpha 
    
    for _ in range(int(probes)):
        i, j = sorted(random.sample(range(n), 2))
        if j - i < 2:
            continue
            
        candidate = np.concatenate((path[:i], path[i:j+1][::-1], path[j+1:]))
        cost = tsp_cost(candidate, dist)
        
        if cost < current_cost:
            current_cost = cost
            best_path = candidate
            improved = True
            break
            
    return best_path, current_cost, improved

# ====================================================================
# 4. كلاس Abstract MAG / Abstract MAG Class
# ====================================================================
class AbstractMAG:
    def __init__(self, problem_data, alpha=100, kick_every=5, time_limit=120.0):
        self.alpha = alpha
        self.kick_every = kick_every
        self.time_limit = time_limit
        
        self.obj_func = tsp_cost 
        self.neighbor = tsp_neighbor_2opt 
        self.kick = tsp_kick 
        self.data = problem_data

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
# 5. التنفيذ الرئيسي / Main Execution
# ====================================================================
if __name__ == "__main__":
    
    # Configuration (Matches the final 120s successful run)
    FILE_PATH = "/content/drive/MyDrive/uy734.tsp"  
    TIME_TO_RUN = 120.0 
    BEST_KNOWN_COST = 79114 

    print("--- 🏁 Running Abstract MAG (120 seconds, Final Settings) 🏁 ---")
    print(f"Loading data from {FILE_PATH}...")
    dist_matrix = load_tsplib_data(FILE_PATH)
    if dist_matrix is None:
        exit()

    N_CITIES = len(dist_matrix)
    print(f"✅ Data loaded: {N_CITIES} cities for Uy734.")

    print("Running Greedy for initial solution...")
    initial = greedy_tsp(dist_matrix)
    initial_cost = tsp_cost(initial, dist_matrix)
    print(f"Greedy cost: {initial_cost:.2f}")

    print(f"Running Abstract MAG with Optimized 2-Opt ({TIME_TO_RUN} seconds)...")
    
    solver = AbstractMAG(dist_matrix, time_limit=TIME_TO_RUN, alpha=100, kick_every=5) 
    best_path, best_cost = solver.solve(initial)

    print(f"\n✅ Abstract MAG finished execution on Uy734")
    print(f"Best Final Cost: {best_cost:.2f}")
    
    gap = ((best_cost - BEST_KNOWN_COST) / BEST_KNOWN_COST) * 100
    print(f"Gap from known optimal ({BEST_KNOWN_COST}): {gap:.2f}%")
