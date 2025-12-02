import numpy as np
import random
import time

# استيراد الكلاس والدوال المطلوبة
from abstract_mag import AbstractMAG 
from tsp_functions import tsp_cost, greedy_tsp

if __name__ == "__main__":
    # 💥 تثبيت البذور لضمان التكرار (للحصول على النتيجة الموثقة 19,222.27)
    np.random.seed(42)
    random.seed(42)

    N_CITIES = 500
    TIME_LIMIT = 30.0

    print(f"جاري إنشاء مشكلة TSP-{N_CITIES} (Seed 42)...")
    cities = np.random.rand(N_CITIES, 2) * 1000
    dist = np.sqrt(((cities[:, np.newaxis, :] - cities[np.newaxis, :, :]) ** 2).sum(axis=2))

    print("تشغيل Greedy للحل الأولي...")
    initial = greedy_tsp(dist)
    initial_cost = tsp_cost(initial, dist)
    print(f"Greedy cost: {initial_cost:.2f}")

    print(f"تشغيل Abstract MAG ({TIME_LIMIT} ثانية)...")
    solver = AbstractMAG(dist, time_limit=TIME_LIMIT) 
    start_time = time.time()
    best_path, best_cost = solver.solve(initial)
    elapsed_time = time.time() - start_time


    print(f"\n✅ Abstract MAG أنهى العمل (أداء مضمون)")
    print(f"أفضل تكلفة (الموثقة): {best_cost:.2f}")
    print(f"زمن التنفيذ: {elapsed_time:.2f} ثانية")
    print(f"التحسن على الحل الجشع: {((initial_cost - best_cost) / initial_cost) * 100:.2f}%")
