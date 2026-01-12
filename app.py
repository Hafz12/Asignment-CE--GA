import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Delhi Metro GA Optimization",
    layout="wide"
)

# ===============================
# Load GA Results
# ===============================
pareto = joblib.load("ga_pareto.pkl")
convergence = joblib.load("ga_convergence.pkl")
runtime = joblib.load("ga_runtime.pkl")

pareto_df = pd.DataFrame(pareto, columns=["Total Distance", "Total Fare"])

# ===============================
# Title
# ===============================
st.title("🧬 Delhi Metro Multi-Objective Optimization (Genetic Algorithm)")

st.markdown("""
This dashboard presents results generated using a  
**Multi-Objective Genetic Algorithm (GA)** to optimize metro routes  
based on **distance** and **fare**.
""")

# ===============================
# Sidebar
# ===============================
st.sidebar.header("⚙️ GA Controls")

solution_idx = st.sidebar.slider(
    "Select Pareto Solution",
    0,
    len(pareto_df) - 1,
    0
)

# ===============================
# Pareto Front
# ===============================
st.subheader("📌 Pareto Front (Distance vs Fare)")

fig, ax = plt.subplots()
ax.scatter(pareto_df["Total Distance"], pareto_df["Total Fare"])
ax.scatter(
    pareto_df.iloc[solution_idx]["Total Distance"],
    pareto_df.iloc[solution_idx]["Total Fare"],
    color="red",
    label="Selected Solution"
)
ax.set_xlabel("Total Distance (Minimize)")
ax.set_ylabel("Total Fare (Minimize)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ===============================
# Selected Solution
# ===============================
st.subheader("🔍 Selected Solution")

c1, c2 = st.columns(2)
c1.metric("Total Distance", f"{pareto_df.iloc[solution_idx]['Total Distance']:.2f} km")
c2.metric("Total Fare", f"₹{pareto_df.iloc[solution_idx]['Total Fare']:.2f}")

# ===============================
# Convergence
# ===============================
st.subheader("📈 Convergence Analysis")

fig2, ax2 = plt.subplots()
ax2.plot(convergence)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Pareto Solutions")
ax2.set_title("GA Pareto Front Growth")
ax2.grid(True)

st.pyplot(fig2)

# ===============================
# Runtime
# ===============================
st.subheader("⏱ Computational Cost")
st.metric("Execution Time", f"{runtime:.3f} seconds")

# ===============================
# Explanation
# ===============================
st.subheader("🧠 Genetic Algorithm Explanation")

st.markdown("""
**GA Mechanism**
- Binary chromosome represents route selection
- Tournament selection preserves good solutions
- Crossover explores new regions
- Mutation avoids premature convergence

**Why GA**
- Strong global search capability
- Handles discrete combinatorial problems
- Produces a well-distributed Pareto front
""")

# ===============================
# Conclusion
# ===============================
st.subheader("✅ Conclusion")

st.markdown("""
The Genetic Algorithm effectively balances competing objectives  
of distance and fare, producing diverse trade-off solutions  
that support informed transportation planning decisions.
""")
