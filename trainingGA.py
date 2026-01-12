import numpy as np
import pandas as pd
import time
import joblib

# =====================================================
# 1. Load Dataset
# =====================================================
data = pd.read_csv("delhi_metro_updated2.0.csv")
data = data[['Distance_km', 'Fare']].dropna()
data = data.head(300)  # control problem size

distance = data["Distance_km"].values
fare = data["Fare"].values
n_dim = len(distance)

# =====================================================
# 2. GA Parameters
# =====================================================
POP_SIZE = 60
N_GENERATIONS = 100
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.02
TOURNAMENT_K = 3

# =====================================================
# 3. Objective Function
# =====================================================
def objectives(solution):
    total_distance = np.sum(solution * distance)
    total_fare = np.sum(solution * fare)
    return np.array([total_distance, total_fare])

# =====================================================
# 4. Pareto Dominance
# =====================================================
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

# =====================================================
# 5. Initialize Population
# =====================================================
population = np.random.randint(0, 2, size=(POP_SIZE, n_dim))

archive = []
convergence = []

start_time = time.time()

# =====================================================
# 6. GA Main Loop
# =====================================================
for gen in range(N_GENERATIONS):

    obj_population = np.array(
        [objectives(ind) for ind in population]
    )

    # -------------------------------
    # Update Pareto Archive
    # -------------------------------
    for i, obj in enumerate(obj_population):
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

    # -------------------------------
    # Tournament Selection
    # -------------------------------
    selected = []
    for _ in range(POP_SIZE):
        candidates = np.random.choice(
            POP_SIZE, TOURNAMENT_K, replace=False
        )
        best = candidates[0]
        for j in candidates[1:]:
            if dominates(
                obj_population[j],
                obj_population[best]
            ):
                best = j
        selected.append(population[best])

    selected = np.array(selected)

    # -------------------------------
    # Crossover
    # -------------------------------
    offspring = []
    for i in range(0, POP_SIZE, 2):
        p1 = selected[i]
        p2 = selected[(i + 1) % POP_SIZE]

        if np.random.rand() < CROSSOVER_RATE:
            cp = np.random.randint(1, n_dim - 1)
            c1 = np.concatenate([p1[:cp], p2[cp:]])
            c2 = np.concatenate([p2[:cp], p1[cp:]])
        else:
            c1 = p1.copy()
            c2 = p2.copy()

        offspring.extend([c1, c2])

    offspring = np.array(offspring[:POP_SIZE])

    # -------------------------------
    # Mutation
    # -------------------------------
    mutation_mask = np.random.rand(
        POP_SIZE, n_dim
    ) < MUTATION_RATE

    population = np.logical_xor(
        offspring, mutation_mask
    ).astype(int)

end_time = time.time()

# =====================================================
# 7. Extract & Save Results
# =====================================================
pareto = [(a[0][0], a[0][1]) for a in archive]

joblib.dump(pareto, "ga_pareto.pkl")
joblib.dump(convergence, "ga_convergence.pkl")
joblib.dump(end_time - start_time, "ga_runtime.pkl")

# =====================================================
# 8. Summary Output
# =====================================================
print("===================================")
print("GENETIC ALGORITHM COMPLETED")
print("Population Size:", POP_SIZE)
print("Generations:", N_GENERATIONS)
print("Pareto Solutions:", len(pareto))
print("Execution Time:", round(end_time - start_time, 3), "seconds")
print("===================================")
