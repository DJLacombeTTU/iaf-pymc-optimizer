import numpy as np
import pymc as pm
import arviz as az
from iaf_optimizer import IAFOptimizer

# ==========================================
# 1. Simulate the Data Generating Process
# ==========================================
np.random.seed(42)
N = 1000

# Generate explanatory variables
log_income = np.random.normal(loc=10.5, scale=0.8, size=N)
financial_literacy = np.random.normal(loc=50.0, scale=10.0, size=N)

# Define true parameters
true_alpha = -5.0
true_beta_income = 1.2
true_beta_lit = 0.3
true_sigma = 2.5

# Generate the continuous dependent variable
mu_true = true_alpha + (true_beta_income * log_income) + (true_beta_lit * financial_literacy)
satisfaction = np.random.normal(loc=mu_true, scale=true_sigma, size=N)

print("Data simulation complete. True Parameters:")
print(f"Alpha: {true_alpha} | Beta(Log_Income): {true_beta_income} | Beta(Lit): {true_beta_lit} | Sigma: {true_sigma}\n")

# ==========================================
# 2. Specify the PyMC Model
# ==========================================
with pm.Model() as satisfaction_model:
    # Uninformative Priors
    alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
    beta_income = pm.Normal("beta_income", mu=0.0, sigma=10.0)
    beta_lit = pm.Normal("beta_lit", mu=0.0, sigma=10.0)
    sigma = pm.HalfNormal("sigma", sigma=5.0)

    # Expected value
    mu_est = alpha + (beta_income * log_income) + (beta_lit * financial_literacy)

    # Likelihood
    Y_obs = pm.Normal("Y_obs", mu=mu_est, sigma=sigma, observed=satisfaction)

# ==========================================
# 3. Train the IAF Universal Approximator
# ==========================================
print("Initializing IAF Optimizer...")
# We use a lightweight configuration since this posterior is highly Gaussian
optimizer = IAFOptimizer(
    pymc_model=satisfaction_model,
    depth=8,                  # Only 2 layers needed for a simple regression
    hidden_sizes=[64, 64],    # Small network 
    num_components=8        # 8 mixture components is plenty here
)

print("Training Flow via ELBO Maximization...")
# We can use a slightly aggressive learning rate for simple geometries
optimizer.fit(learning_rate=0.01, num_steps=3000, batch_size=256)

# ==========================================
# 4. Generate Inference Data and Verify
# ==========================================
print("\nSampling from optimized flow...")
idata = optimizer.sample(num_samples=2000)

print("\n--- Parameter Recovery Results ---")
summary_table = optimizer.summary(idata)
print(summary_table)


