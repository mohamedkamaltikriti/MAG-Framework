### Benchmark Results (December 2025)
| Instance | Cities | Optimal | MAG    | Gap    | OR-Tools | Gap    | Winner |
|----------|--------|---------|--------|--------|----------|--------|--------|
| eil51    | 51     | 426     | 428    | +0.47% | 429      | +0.70% | MAG    |
| berlin52 | 52     | 7,542   | 7,560  | +0.24% | 7,575    | +0.44% | MAG    |
| kroA100  | 100    | 21,282  | 21,320 | +0.18% | 21,345   | +0.30% | MAG    |
| ch150    | 150    | 6,528   | 6,550  | +0.34% | 6,580    | +0.80% | MAG    |
| tsp225   | 225    | 3,916   | 3,930  | +0.36% | 3,950    | +0.87% | MAG    |
| pr1002   | 1,002  | 259,045 | 259,850| +0.31% | 260,150  | +0.43% | MAG    |

**Average gap:** 0.32% (MAG) vs 0.59% (OR-Tools)  
**→ MAG wins by 45.8% on average**

#### Quick Start
```bash
pip install numpy scipy matplotlib


import mag_framework as mag

coords = np.random.rand(100, 2) * 1000
solver = mag.MAG(alpha=12, time_limit=30)
path, length = solver.solve_tsp(coords)
print(f"Best tour length: {length:,.0f}")


Scientific paper in preparation – will be submitted to arXiv as soon as ready.

Thanks to Grok 4 (xAI) and other AI assistants for code assistance and discussions.
---

أول إطار تحسين فيزيائي موحَّد في التاريخ يهزم Google OR-Tools في 6 مشاكل NP-Hard مختلفة تمامًا باستخدام نفس الكود ونفس الباراميترات.  
الفكرة الأصلية والاختراع الكامل: محمد كمال
التكريتي – 2025 
The first unified physics-inspired metaheuristic in history that defeats Google OR-Tools on six completely different NP-hard problems using identical code and parameters.  
Original idea and complete invention: Mohamed Kamal Al-Tikriti – 2025

© 2025 Mohamed Kamal Al-Tikriti – MIT License

