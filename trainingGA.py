import numpy as np
import pandas as pd
import time
import joblib

# ===============================
# 1. Load Dataset
# ===============================
data = pd.read_csv("delhi_metro_updated2.0 (1).csv")
data = data.head(300)

distance = data["Distance_km"].values
fare = data["Fare"].values
n_dim = len(distance)

# ===============================
# 2. GA Parameters
# ===============================
pop_size = 60
n_generations = 100
crossover_rate = 0.9
mutation_rate = 0.02
tournament_k = 3

# ===============================
# 3. Objective Function
# ===============================
def objectives(solution):
    return np.array([
        np.sum(solution * distance),
        np.sum(solution * fare)
    ])

# ===============================
# 4. Dominance Check
# ===============================
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

# ===============================
# 5. Initialize Population
# ===============================
population = np.random.randint(0, 2, size=(pop_size, n_dim))

archive = []
convergence = []

start_time = time.time()

# ===============================
# 6. GA Main Loop
# ===============================
for gen in range(n_generations):
    objectives_pop = np.array([objectives(ind) for ind in population])

    # --- Pareto Archive Update ---
    for i, obj in enumerate(objectives_pop):
        dominated = False
        new_archive = []

        for a in archive:
            if dominates(a[0], obj):
                dominated = True
                break
            if not dominates(obj, a[0]):
                new_archive.append(a)

        if not dominated:
            new_archive.append((obj, population[i]))
            archive = new_archive

    convergence.append(len(archive))

    # --- Tournament Selection ---
    selected = []
    for _ in range(pop_size):
        idx = np.random.choice(pop_size, tournament_k, replace=False)
        best = idx[0]
        for j in idx[1:]:
            if dominates(objectives_pop[j], objectives_pop[best]):
                best = j
        selected.append(population[best])
    selected = np.array(selected)

    # --- Crossover ---
    offspring = []
    for i in range(0, pop_size, 2):
        p1, p2 = selected[i], selected[(i + 1) % pop_size]
        if np.random.rand() < crossover_rate:
            cp = np.random.randint(1, n_dim - 1)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
        else:
            c1, c2 = p1.copy(), p2.copy()
        offspring.extend([c1, c2])

    offspring = np.array(offspring[:pop_size])

    # --- Mutation ---
    mutation_mask = np.random.rand(pop_size, n_dim) < mutation_rate
    offspring = np.logical_xor(offspring, mutation_mask).astype(int)

    population = offspring

end_time = time.time()

# ===============================
# 7. Save Results
# ===============================
pareto = [(a[0][0], a[0][1]) for a in archive]

joblib.dump(pareto, "ga_pareto.pkl")
joblib.dump(convergence, "ga_convergence.pkl")
joblib.dump(end_time - start_time, "ga_runtime.pkl")

print("GA optimization completed and saved.")
