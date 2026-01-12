import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Delhi Metro Route Optimization (GA)",
    layout="wide"
)

# =====================================================
# Load Dataset
# =====================================================
data = pd.read_csv("delhi_metro_updated2.0 (2).csv")
data = data[['Distance_km', 'Fare']].dropna()

distance = data["Distance_km"].values
fare = data["Fare"].values
n_dim = len(distance)

# =====================================================
# GA Functions
# =====================================================
def objectives(solution):
    return np.array([
        np.sum(solution * distance),
        np.sum(solution * fare)
    ])

def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

def run_ga(pop_size, n_gen, crossover_rate, mutation_rate):
    population = np.random.randint(0, 2, size=(pop_size, n_dim))
    archive = []
    convergence = []

    start_time = time.time()

    for gen in range(n_gen):
        obj_pop = np.array([objectives(ind) for ind in population])

        # Pareto archive update
        for i, obj in enumerate(obj_pop):
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

        # Tournament selection
        selected = []
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, 2, replace=False)
            winner = i if dominates(obj_pop[i], obj_pop[j]) else j
            selected.append(population[winner])
        selected = np.array(selected)

        # Crossover
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

        # Mutation
        mutation_mask = np.random.rand(pop_size, n_dim) < mutation_rate
        population = np.logical_xor(offspring, mutation_mask).astype(int)

    runtime = time.time() - start_time
    pareto = [a[0] for a in archive]

    return pareto, convergence, runtime

# =====================================================
# Title and Description
# =====================================================
st.title("🧬 Delhi Metro Route Optimization (Genetic Algorithm)")

st.markdown("""
This application allows users to interactively tune  
**Genetic Algorithm (GA) parameters** to optimize metro routes  
based on **distance and fare minimization**.
""")

# =====================================================
# Sidebar – GA Parameters
# =====================================================
st.sidebar.header("⚙️ GA Parameters")

pop_size = st.sidebar.slider("Population Size", 20, 200, 60, step=10)
n_gen = st.sidebar.slider("Generations", 20, 300, 100, step=20)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.5, 1.0, 0.9)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.001, 0.1, 0.02)

run_button = st.sidebar.button("🚀 Run GA")

# =====================================================
# Run GA
# =====================================================
if run_button:
    with st.spinner("Running Genetic Algorithm..."):
        pareto, convergence, runtime = run_ga(
            pop_size, n_gen, crossover_rate, mutation_rate
        )

    if len(pareto) == 0:
        st.error("No Pareto solutions found. Try increasing generations.")
        st.stop()

    pareto_df = pd.DataFrame(
        pareto, columns=["Total Distance", "Total Fare"]
    )

    # =====================================================
    # Best GA Optimized Solution
    # =====================================================
    st.subheader("🏆 Best GA Optimized Solutions")

    best_idx = np.argmin(pareto_df["Total Distance"])
    best = pareto_df.iloc[best_idx]

    b1, b2, b3 = st.columns(3)
    b1.metric("Min Distance (km)", f"{best['Total Distance']:.2f}")
    b2.metric("Min Fare (₹)", f"{best['Total Fare']:.2f}")
    b3.metric("Runtime (s)", f"{runtime:.3f}")

    # =====================================================
    # Pareto Front
    # =====================================================
    st.subheader("📌 Pareto Front")

    fig, ax = plt.subplots()
    ax.scatter(
        pareto_df["Total Distance"],
        pareto_df["Total Fare"]
    )
    ax.set_xlabel("Total Distance (Minimize)")
    ax.set_ylabel("Total Fare (Minimize)")
    ax.grid(True)

    st.pyplot(fig)

    # =====================================================
    # Convergence
    # =====================================================
    st.subheader("📈 Convergence Analysis")

    fig2, ax2 = plt.subplots()
    ax2.plot(convergence)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Pareto Solutions")
    ax2.grid(True)

    st.pyplot(fig2)

# =====================================================
# Explanation
# =====================================================
with st.expander("🧠 GA Parameter Interpretation"):
    st.markdown("""
    - **Population Size** controls solution diversity
    - **Generations** determine search depth
    - **Crossover Rate** enables exploration
    - **Mutation Rate** prevents premature convergence
    """)

# =====================================================
# Conclusion
# =====================================================
st.subheader("✅ Conclusion")

st.markdown("""
Interactive parameter tuning allows users to observe  
how GA hyperparameters affect convergence behavior  
and solution quality in multi-objective optimization.
""")
