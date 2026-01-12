import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Delhi Metro Route Optimization (GA)",
    layout="wide"
)

# =====================================================
# Load GA Results
# =====================================================
pareto = joblib.load("ga_pareto.pkl")
convergence = joblib.load("ga_convergence.pkl")
runtime = joblib.load("ga_runtime.pkl")

pareto_df = pd.DataFrame(
    pareto, columns=["Total Distance", "Total Fare"]
)

# =====================================================
# Load Dataset
# =====================================================
data = pd.read_csv("delhi_metro_updated2.0.csv")
data = data[['Distance_km', 'Fare']].dropna()

# =====================================================
# Title and Description
# =====================================================
st.title("🧬 Delhi Metro Route Optimization (Genetic Algorithm)")

st.markdown(
    """
    This application optimizes **Delhi Metro route selection** using a  
    **Multi-Objective Genetic Algorithm (GA)**.

    The GA simultaneously minimizes **total travel distance** and  
    **total fare cost**, producing a set of **Pareto-optimal solutions**
    that represent optimal trade-offs between conflicting objectives.
    """
)

# =====================================================
# Sidebar – Solution Selection
# =====================================================
st.sidebar.header("🧩 Pareto Solution Selector")

solution_idx = st.sidebar.slider(
    "Select Pareto Solution",
    min_value=0,
    max_value=len(pareto_df) - 1,
    value=0
)

selected_solution = pareto_df.iloc[solution_idx]

# =====================================================
# Best GA Optimized Solution
# =====================================================
st.subheader("🏆 Best GA Optimized Solution (Selected Pareto Point)")

gcol1, gcol2, gcol3 = st.columns(3)

gcol1.metric(
    "Total Distance (km)",
    f"{selected_solution['Total Distance']:.2f}"
)

gcol2.metric(
    "Total Fare (₹)",
    f"{selected_solution['Total Fare']:.2f}"
)

gcol3.metric(
    "Execution Time (s)",
    f"{runtime:.3f}"
)

st.caption(
    "The selected solution represents a **Pareto-optimal route configuration** "
    "identified by the Genetic Algorithm."
)

# =====================================================
# Pareto Front Visualization
# =====================================================
st.subheader("📌 Pareto Front (Distance vs Fare)")

fig, ax = plt.subplots()
ax.scatter(
    pareto_df["Total Distance"],
    pareto_df["Total Fare"],
    label="Pareto Solutions"
)

ax.scatter(
    selected_solution["Total Distance"],
    selected_solution["Total Fare"],
    color="red",
    label="Selected Solution"
)

ax.set_xlabel("Total Distance (Minimize)")
ax.set_ylabel("Total Fare (Minimize)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# =====================================================
# Convergence Analysis
# =====================================================
st.subheader("📈 Convergence Analysis")

fig2, ax2 = plt.subplots()
ax2.plot(convergence)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Number of Pareto Solutions")
ax2.set_title("GA Pareto Front Growth")
ax2.grid(True)

st.pyplot(fig2)

# =====================================================
# Dataset Preview
# =====================================================
st.subheader("🗂 Dataset Preview")

with st.expander("Show dataset sample"):
    st.dataframe(data.head(10))

# =====================================================
# GA Explanation
# =====================================================
with st.expander("🧠 How Genetic Algorithm Works"):
    st.markdown(
        """
        - Each chromosome represents a **binary route selection vector**
        - Fitness evaluation is based on **distance and fare minimization**
        - **Tournament selection** chooses fitter parents
        - **Crossover** explores new route combinations
        - **Mutation** prevents premature convergence
        - Pareto dominance ensures **solution diversity**
        """
    )

# =====================================================
# Conclusion
# =====================================================
st.subheader("✅ Conclusion")

st.markdown(
    """
    The Genetic Algorithm effectively identifies optimal trade-offs  
    between travel distance and fare cost for Delhi Metro routes.

    By visualizing Pareto fronts and convergence behavior,  
    this system enhances **interpretability and decision support**  
    for multi-objective transportation planning.
    """
)
