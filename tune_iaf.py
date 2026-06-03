import numpy as np
import pymc as pm
import optuna
from arviz import psislw
from iaf_optimizer import IAFOptimizer

# ==========================================
# 1. Setup the Exact Same Model
# ==========================================
np.random.seed(42)
N = 1000
log_income = np.random.normal(loc=10.5, scale=0.8, size=N)
financial_literacy = np.random.normal(loc=50.0, scale=10.0, size=N)
satisfaction = np.random.normal(
    loc=-5.0 + (1.2 * log_income) + (0.3 * financial_literacy), 
    scale=2.5, size=N
)

with pm.Model() as satisfaction_model:
    alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
    beta_income = pm.Normal("beta_income", mu=0.0, sigma=10.0)
    beta_lit = pm.Normal("beta_lit", mu=0.0, sigma=10.0)
    sigma = pm.HalfNormal("sigma", sigma=5.0)
    mu_est = alpha + (beta_income * log_income) + (beta_lit * financial_literacy)
    Y_obs = pm.Normal("Y_obs", mu=mu_est, sigma=sigma, observed=satisfaction)

# ==========================================
# 2. Define the Optuna Objective
# ==========================================
def objective(trial):
    """
    Optuna will run this function dozens of times, testing different parameters.
    """
    # Define the Search Space
    depth = trial.suggest_int("depth", 2, 8, step=2)
    num_components = trial.suggest_int("num_components", 8, 32, step=8)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128])
    hidden_sizes = [hidden_dim, hidden_dim]

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: depth={depth}, components={num_components}, lr={lr:.5f}, batch={batch_size}")

    # Initialize the flow with trial parameters
    optimizer = IAFOptimizer(
        pymc_model=satisfaction_model,
        depth=depth,
        hidden_sizes=hidden_sizes,
        num_components=num_components
    )

    # Define the Pruning Callback
    def pruning_callback(step, loss):
        trial.report(loss, step)
        # Check if the trial is performing poorly compared to previous ones
        if trial.should_prune():
            raise optuna.TrialPruned()
        return False

    # Train the flow
    optimizer.fit(
        learning_rate=lr, 
        num_steps=2000, 
        batch_size=batch_size, 
        callback=pruning_callback
    )
    
    # Calculate Pareto k to evaluate posterior coverage
    idata = optimizer.sample(num_samples=2000)
    log_weights = idata.sample_stats["log_weights"].values.flatten()
    _, pareto_k = psislw(log_weights)
    trial.set_user_attr("pareto_k", float(pareto_k))
    
    # Evaluate geometric coverage and penalize bad shapes
    final_loss = optimizer.final_loss
    
    # Define a severe penalty for failing the Pareto k diagnostic
    penalty = 0.0
    if pareto_k > 0.7:
        penalty = 10000.0 * pareto_k  # Hard fail: Force Optuna to abandon this region
    elif pareto_k > 0.5:
        penalty = 1000.0 * pareto_k   # Soft warning: Discourage but keep alive
        
    # Return the penalized loss to Optuna
    penalized_loss = final_loss + penalty
    
    # Store the pure loss for your records
    trial.set_user_attr("pure_loss", float(final_loss))
    
    return penalized_loss

# ==========================================
# 3. Execute the Study
# ==========================================
if __name__ == "__main__":
    # We use the MedianPruner: Kills trials that perform worse than the median of past trials
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=500)
    
    # Save results to a local SQLite database so you can pause/resume tuning
    study = optuna.create_study(
        study_name="iaf_satisfaction_tuning", 
        storage="sqlite:///iaf_tuning.db", 
        load_if_exists=True,
        direction="minimize",
        pruner=pruner
    )
    
    print("Beginning hyperparameter optimization...")
    # Run 30 sequential trials (N1 strategy to prevent GPU memory crashes)
    study.optimize(objective, n_trials=30)
    
    print("\n==================================")
    print(" OPTIMIZATION COMPLETE ")
    print("==================================")
    print("Best Trial:")
    best_trial = study.best_trial
    print(f"  Penalized Loss: {best_trial.value:.4f}")
    
    pure_loss = best_trial.user_attrs.get('pure_loss')
    print(f"  Pure Loss:      {pure_loss:.4f}" if pure_loss is not None else "  Pure Loss:      N/A")
    
    pareto_k = best_trial.user_attrs.get('pareto_k')
    print(f"  Pareto k:       {pareto_k:.4f}" if pareto_k is not None else "  Pareto k:       N/A")
    
    # ... existing print statements ...
    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- ADD THIS NEW SECTION ---
    print("\n" + "="*65)
    print(" OPTIMIZED CODE SNIPPET (Ready to Copy & Paste)")
    print("="*65)
    
    depth = best_trial.params['depth']
    hidden_dim = best_trial.params['hidden_dim']
    num_components = best_trial.params['num_components']
    lr = best_trial.params['learning_rate']
    batch = best_trial.params['batch_size']
    
    code_snippet = f"""
# ==========================================
# 3. Train the IAF Universal Approximator
# ==========================================
print("Initializing Optimized IAF Architecture...")
optimizer = IAFOptimizer(
    pymc_model=your_pymc_model, # <-- Update this if not 'satisfaction_model'
    depth={depth},                  
    hidden_sizes=[{hidden_dim}, {hidden_dim}],  
    num_components={num_components}         
)

print("Training Flow via ELBO Maximization...")
optimizer.fit(learning_rate={lr:.7f}, num_steps=3000, batch_size={batch})
"""
    print(code_snippet)
    print("="*65 + "\n")