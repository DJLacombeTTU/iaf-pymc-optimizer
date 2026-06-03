import numpy as np
import pymc as pm
import optuna
from arviz import psislw
from iaf_optimizer import IAFOptimizer

# ==========================================
# 1. The Core Tuning and Evaluation Engine
# ==========================================
def tune_and_estimate(model_name: str, pymc_model: pm.Model, n_trials: int = 20, iwae_k: int = 1):
    print("\n" + "="*70)
    print(f" BENCHMARKING: {model_name.upper()}")
    print("="*70)
    if iwae_k > 1:
        print(f"[*] Using IWAE Objective with k={iwae_k} for mass-covering geometry.")
    else:
        print("[*] Using Standard ELBO Objective.")

    # --- Phase A: Optuna Hyperparameter Tuning ---
    def objective(trial):
        depth = trial.suggest_int("depth", 2, 8, step=2)
        num_components = trial.suggest_int("num_components", 8, 32, step=8)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128])
        hidden_sizes = [hidden_dim, hidden_dim]

        optimizer = IAFOptimizer(
            pymc_model=pymc_model,
            depth=depth,
            hidden_sizes=hidden_sizes,
            num_components=num_components
        )

        def pruning_callback(step, loss):
            trial.report(loss, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return False

        # Train with fewer steps during the search phase to save compute
        optimizer.fit(
            learning_rate=lr, 
            num_steps=1000, 
            batch_size=batch_size, 
            iwae_k=iwae_k, 
            callback=pruning_callback
        )
        
        # Calculate Pareto k penalty
        idata = optimizer.sample(num_samples=1000)
        log_weights = idata.sample_stats["log_weights"].values.flatten()
        _, pareto_k = psislw(log_weights)
        trial.set_user_attr("pareto_k", float(pareto_k))
        
        penalty = 0.0
        if pareto_k > 0.7:
            penalty = 10000.0 * pareto_k
        elif pareto_k > 0.5:
            penalty = 1000.0 * pareto_k
            
        return optimizer.final_loss + penalty

    # Run the study
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=300)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    print(f"\n--- Launching Optuna Search ({n_trials} trials) ---")
    
    # Temporarily suppress Optuna's verbose logging to keep the console clean
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    study.optimize(objective, n_trials=n_trials)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    best = study.best_trial
    print(f"\n[Search Complete] Optimal Geometry Found:")
    print(f"Depth: {best.params['depth']} | Components: {best.params['num_components']} | Hidden: {best.params['hidden_dim']}")
    print(f"LR: {best.params['learning_rate']:.6f} | Batch: {best.params['batch_size']}")

    # --- Phase B: Final Estimation and Summary ---
    print("\n--- Estimating Final Model with Optimal Parameters ---")
    final_optimizer = IAFOptimizer(
        pymc_model=pymc_model,
        depth=best.params['depth'],
        hidden_sizes=[best.params['hidden_dim'], best.params['hidden_dim']],
        num_components=best.params['num_components']
    )

    # Train for a full 3000 steps for the final estimation
    final_optimizer.fit(
        learning_rate=best.params['learning_rate'], 
        num_steps=3000, 
        batch_size=best.params['batch_size'],
        iwae_k=iwae_k
    )

    print("\n--- Generating Posterior Samples ---")
    final_idata = final_optimizer.sample(num_samples=5000)
    
    print(f"\n[ Final ArviZ Summary for {model_name} ]")
    summary_df = final_optimizer.summary(final_idata)
    print(summary_df)
    return final_optimizer, final_idata

# ==========================================
# 2. Define the Benchmark Models
# ==========================================

def build_centered_funnel():
    """Test 1: The Centered Funnel (High Curvature Gradients)"""
    np.random.seed(42)
    J = 50 
    true_mu = 8.0
    true_tau = 3.0
    true_theta = np.random.normal(true_mu, true_tau, J)
    sigma = np.random.uniform(1, 5, J)
    y_obs = np.random.normal(true_theta, sigma)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        tau = pm.HalfNormal("tau", sigma=10)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)
        pm.Normal("obs", mu=theta, sigma=sigma, observed=y_obs)
    return model

def build_dense_correlation():
    """Test 2: Dense High-Dimensional Correlations"""
    np.random.seed(42)
    D = 100 # Kept at 100 so tuning runs reasonably fast for the demo
    cov_matrix = np.exp(-0.1 * np.abs(np.subtract.outer(np.arange(D), np.arange(D))))
    true_L = np.linalg.cholesky(cov_matrix)
    true_beta = true_L @ np.random.normal(size=D)
    X = np.random.normal(size=(500, D))
    y_obs = X @ true_beta + np.random.normal(size=500)

    with pm.Model() as model:
        sd = pm.HalfNormal("sd", sigma=5)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=D)
        mu_est = pm.math.dot(X, beta)
        pm.Normal("obs", mu=mu_est, sigma=sd, observed=y_obs)
    return model

def build_massive_logistic():
    """Test 3: Large Dataset Evaluation"""
    np.random.seed(42)
    N_massive = 100_000 # 100k rows to showcase JAX vectorization speed
    X_massive = np.random.normal(size=(N_massive, 3))
    true_weights = np.array([1.5, -2.0, 0.5])
    logits = X_massive @ true_weights
    p = 1 / (1 + np.exp(-logits))
    y_massive = np.random.binomial(1, p)

    with pm.Model() as model:
        weights = pm.Normal("weights", mu=0, sigma=5, shape=3)
        mu_est = pm.math.dot(X_massive, weights)
        pm.Bernoulli("obs", logit_p=mu_est, observed=y_massive)
    return model

def build_uncentered_funnel():
    """Test 4: The Uncentered Funnel (Isotropic Gaussian Geometry)"""
    np.random.seed(42)
    J = 50 
    true_mu = 8.0
    true_tau = 3.0
    true_theta = np.random.normal(true_mu, true_tau, J)
    sigma = np.random.uniform(1, 5, J)
    y_obs = np.random.normal(true_theta, sigma)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)
        tau = pm.HalfNormal("tau", sigma=10)
        
        # The Non-Centered Parameterization
        theta_raw = pm.Normal("theta_raw", mu=0, sigma=1, shape=J)
        
        # Deterministic transformation maps it back to the true scale
        theta = pm.Deterministic("theta", mu + theta_raw * tau)
        
        pm.Normal("obs", mu=theta, sigma=sigma, observed=y_obs)
    return model

# ==========================================
# 3. Execute the Benchmarks
# ==========================================
if __name__ == "__main__":
    
    # Test 1: Centered Funnel (Using IWAE k=16 to force tail expansion)
    m1 = build_centered_funnel()
    tune_and_estimate("Centered Funnel of Hell", m1, n_trials=15, iwae_k=16)
    
    # Test 2: Dense Correlation (Standard ELBO)
    m2 = build_dense_correlation()
    tune_and_estimate("Dense High-Dimensional Correlation", m2, n_trials=15)
    
    # Test 3: Massive Logistic (Standard ELBO)
    m3 = build_massive_logistic()
    tune_and_estimate("Massive Vectorized Logistic", m3, n_trials=15)

    # Test 4: Uncentered Funnel (Standard ELBO - Proving Geometry > Objective)
    m4 = build_uncentered_funnel()
    tune_and_estimate("Uncentered Funnel (Isotropic)", m4, n_trials=15)