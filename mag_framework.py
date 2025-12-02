# MAG-Framework v1.0 - Magnetic Attraction-Repulsion General Optimizer
# Author: Mohamed Kamal Al-Tikriti
# First public release: 1 December 2025
# License: GNU GPL v3 (see LICENSE) | Commercial licensing: ShamsPhone.sh@gmail.com

import numpy as np
from scipy.spatial import cKDTree
import random
import time
from typing import Tuple

class MAG:
    def __init__(self, alpha: int = 12, kick_every: int = 15, time_limit: float = 60.0):
        self.alpha = alpha
        self.kick_every = kick_every
        self.time_limit = time_limit

    def _dist_matrix(self, coords: np.ndarray) -> np.ndarray:
        return np.sqrt(((coords[:, np.newaxis] - coords) ** 2).sum(axis=2))

    def _tour_length(self, path: np.ndarray, dist_matrix: np.ndarray) -> float:
        return dist_matrix[path, np.roll(path, -1)].sum()

    def _build_candidates(self, coords: np.ndarray):
        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=self.alpha + 1)
        return [set(neighbors[1:]) for neighbors in idx]

    def _double_bridge(self, path: np.ndarray) -> np.ndarray:
        n = len(path)
        if n < 12:
            return path.copy()
        a, b, c, d = sorted(random.sample(range(1, n), 4))
        return np.concatenate((path[:a], path[c:d], path[b:c], path[a:b], path[d:]))

    def solve_tsp(self, coords: np.ndarray) -> Tuple[np.ndarray, float]:
        coords = np.asarray(coords, dtype=float)
        n = len(coords)
        dist_matrix = self._dist_matrix(coords)
        
        path = np.random.permutation(n)
        best_path = path.copy()
        best_length = self._tour_length(path, dist_matrix)
        
        candidates = self._build_candidates(coords)
        start_time = time.time()
        kick_counter = 0

        while time.time() - start_time < self.time_limit:
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    a, b = path[i], path[(i + 1) % n]
                    for j_idx in candidates[a]:
                        c, d = path[j_idx], path[(j_idx + 1) % n]
                        delta = (dist_matrix[a,c] + dist_matrix[b,d] 
                               - dist_matrix[a,b] - dist_matrix[c,d])
                        if delta < -1e-8:
                            path[i+1:j_idx+1] = path[i+1:j_idx+1][::-1]
                            improved = True
                            break
                    if improved:
                        break

            length = self._tour_length(path, dist_matrix)
            if length < best_length:
                best_length = length
                best_path = path.copy()

            kick_counter += 1
            if kick_counter >= self.kick_every:
                path = self._double_bridge(best_path)
                kick_counter = 0

        return best_path, best_length
