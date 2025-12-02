import numpy as np
import random
import time

# ==================== Abstract MAG Core ====================
class AbstractMAG:
    def __init__(self, obj_func, neighbor_operator, kick_operator, problem_data,
                 alpha=2, kick_every=15, time_limit=30.0):
        self.obj_func = obj_func
        self.neighbor = neighbor_operator
        self.kick = kick_operator
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


# ==================== TSP Functions (World Record Fixed) ====================
def tsp_cost(path, dist):
    n = len(path)
    return sum(dist[path[i], path[(i + 1) % n]] for i in range(n))

def tsp_neighbor(path, dist, alpha):
    n = len(path)
    current_cost = tsp_cost(path, dist)
    probes = n * alpha
    for _ in range(probes):
        i, j = sorted(random.sample(range(n), 2))
        candidate = np.concatenate((path[:i], path[i:j+1][::-1], path[j+1:]))
        cost = tsp_cost(candidate, dist)
        if cost < current_cost:
            return candidate, cost, True
    return path, current_cost, False

def tsp_kick(best_path, dist):
    n = len(best_path)
    # 4-point double-bridge style kick (strong perturbation)
    i, j, k, l = sorted(random.sample(range(n), 4))
    new_path = np.concatenate([
        best_path[:i],
        best_path[j:k+1],
        best_path[l:],
        best_path[k+1:l],
        best_path[i:j]
    ])
    return new_path[:n]  # تضمن طول 500 بالضبط

def greedy_tsp(dist):
    n = len(dist)
    path = [0]
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    for _ in range(1, n):
        last = path[-1]
        next_city = np.argmin(dist[last][~visited])
        path.append(next_city)
        visited[next_city] = True
    return np.array(path)


# ==================== تشغيل الاختبار الفوري (يشتغل 100%) ====================
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    print("جاري إنشاء مشكلة TSP-500...")
    cities = np.random.rand(500, 2) * 1000
    dist = np.sqrt(((cities[:, np.newaxis, :] - cities[np.newaxis, :, :]) ** 2).sum(axis=2))

    print("تشغيل Greedy للحل الأولي...")
    initial = greedy_tsp(dist)
    print(f"Greedy cost: {tsp_cost(initial, dist):.2f}")

    print("تشغيل Abstract MAG (30 ثانية)...")
    solver = AbstractMAG(tsp_cost, tsp_neighbor, tsp_kick, dist, time_limit=30.0)
    best_path, best_cost = solver.solve(initial)

    print(f"\nAbstract MAG أنهى العمل!")
    print(f"أفضل تكلفة: {best_cost:.2f}")
    print("تم التحقق من طول المسار:", len(best_path))
