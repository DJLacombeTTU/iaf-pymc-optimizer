import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import pandas as pd
from arviz import psislw
from typing import List, Tuple
import optax
import pymc as pm
from pymc.sampling.jax import get_jaxified_logp
import arviz as az
from jax.tree_util import tree_map

class MaskedDense(eqx.Module):
    """
    A Dense layer that enforces an autoregressive structure 
    by multiplying weights by a binary adjacency mask.
    """
    weight: jnp.ndarray
    bias: jnp.ndarray
    mask: jnp.ndarray
    has_bias: bool = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, mask: jnp.ndarray, has_bias: bool = True, zero_init: bool = False, *, key):
        wkey, bkey = jr.split(key)
        
        # --- IDENTITY INITIALIZATION ---
        # If zero_init is True, initialize weights and biases to absolute zero.
        # This ensures the flow starts as an exact Identity function (y=x).
        if zero_init:
            self.weight = jnp.zeros((out_features, in_features))
            if has_bias:
                self.bias = jnp.zeros((out_features,))
            else:
                self.bias = jnp.zeros((out_features,))
        else:
            # Standard Kaiming/Lecun hidden initialization
            lim = 1.0 / jnp.sqrt(in_features)
            self.weight = jr.uniform(wkey, (out_features, in_features), minval=-lim, maxval=lim)
            
            if has_bias:
                self.bias = jr.uniform(bkey, (out_features,), minval=-lim, maxval=lim)
            else:
                self.bias = jnp.zeros((out_features,))
            
        self.mask = mask
        self.has_bias = has_bias

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Enforce the autoregressive constraint dynamically during the forward pass
        masked_weight = self.weight * self.mask
        out = jnp.dot(masked_weight, x)
        if self.has_bias:
            out = out + self.bias
        return out


class AutoregressiveConditioner(eqx.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE) style network.
    Outputs the shifting, scaling, and deep sigmoidal parameters for the flow.
    """
    layers: List[MaskedDense]
    out_features_per_dim: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, hidden_sizes: List[int], out_features_per_dim: int, *, key):
        self.dim = dim
        self.out_features_per_dim = out_features_per_dim
        
        # Assign order degrees to input, hidden, and output units to construct masks
        input_degrees = jnp.arange(dim)
        
        # Hidden degrees are sampled or distributed evenly to ensure valid pathways
        hidden_degrees = []
        current_deg = input_degrees
        
        keys = jr.split(key, len(hidden_sizes) + 1)
        layers = []
        
        # Structural design of hidden masks
        in_features = dim
        for i, h_size in enumerate(hidden_sizes):
            # Assign degrees to hidden nodes sequentially or randomly within [0, dim - 2]
            h_deg = jnp.mod(jnp.arange(h_size), dim - 1)
            hidden_degrees.append(h_deg)
            
            # Mask criteria for hidden layers: hidden_node >= input_node
            mask = h_deg[:, None] >= current_deg[None, :]
            layers.append(MaskedDense(in_features, h_size, mask, key=keys[i]))
            
            in_features = h_size
            current_deg = h_deg

        # Output layer maps to multiple parameters per original input dimension
        # (e.g., location, log-scale, and sigmoidal weights)
        out_features = dim * out_features_per_dim
        output_degrees = jnp.repeat(input_degrees, out_features_per_dim)
        
        # Mask criteria for output layer: output_node > hidden_node (Strictly Autoregressive)
        # We pass zero_init=True to guarantee a safe start for the flow.
        final_mask = output_degrees[:, None] > current_deg[None, :]
        layers.append(MaskedDense(in_features, out_features, final_mask, zero_init=True, key=keys[-1]))
        
        self.layers = layers

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Pass elements sequentially through the masked network hierarchy
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))  # GELU: Smoother manifold mapping
        
        # Final layer remains raw unconstrained logits/parameters
        return self.layers[-1](x)
    
class DeepSigmoidalBijector(eqx.Module):
    """
    Applies a monotonic neural transformation using a mixture of sigmoids.
    Maps R -> R while computing the log-determinant of the Jacobian analytically.
    """
    num_components: int = eqx.field(static=True)

    def __init__(self, num_components: int = 16):
        self.num_components = num_components

    def __call__(self, x: jnp.ndarray, params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x: [D] - The input variables (e.g., from the base distribution).
        params: [D * (3 * num_components)] - The output from the AutoregressiveConditioner.
        
        Returns:
            y: [D] - The transformed variables.
            log_det: [D] - The log absolute derivative of the transformation per dimension.
        """
        D = x.shape[0]
        
        # Reshape the flat parameter vector into the structural components
        # Shape: [D, 3, num_components]
        params = params.reshape(D, 3, self.num_components)
        
        # Extract the specific parameters
        logits = params[:, 0, :]   # Unnormalized mixture weights
        a_raw = params[:, 1, :]    # Pre-activations for scaling
        b = params[:, 2, :]        # Shifts
        
        # Ensure weights sum to 1 across the components
        pi = jax.nn.softmax(logits, axis=-1)
        
        # Ensure 'a' is strictly positive for monotonicity
        a = jax.nn.softplus(a_raw)
        
        # Expand x to broadcast against the mixture components
        # Shape: [D, 1]
        x_expanded = x[:, None]
        
        # Compute the inner sigmoid mixture: u = sum(pi * sigmoid(a * x + b))
        # This maps the real line to the (0, 1) interval
        inner_arg = a * x_expanded + b
        sig_inner = jax.nn.sigmoid(inner_arg)
        u = jnp.sum(pi * sig_inner, axis=-1)
        
        # Numerical stability: clip u to prevent log(0) in the inverse transform
        eps = 1e-6
        u = jnp.clip(u, eps, 1.0 - eps)
        
        # Transform back to the real line using the logit function
        # y = logit(u) = log(u) - log(1 - u)
        y = jax.scipy.special.logit(u)
        
        # --- Analytical Log-Det Jacobian Computation ---
        # 1. Derivative of the mixture sum w.r.t x
        # d(sig)/dx = a * sig * (1 - sig)
        dsig_dx = a * sig_inner * (1.0 - sig_inner)
        du_dx = jnp.sum(pi * dsig_dx, axis=-1)
        
        # 2. Derivative of the logit function w.r.t u
        # dy/du = 1 / (u * (1 - u))
        dy_du = 1.0 / (u * (1.0 - u))
        
        # 3. Chain rule: dy/dx = (dy/du) * (du_dx)
        # log|dy/dx| = log(dy_du) + log(du_dx)
        log_det = jnp.log(dy_du + eps) + jnp.log(du_dx + eps)
        
        return y, log_det

class FlowLayer(eqx.Module):
    """
    A single layer of the Inverse Autoregressive Flow.
    Combines the Conditioner, the Bijector, and a fixed random Permutation.
    """
    conditioner: AutoregressiveConditioner
    bijector: DeepSigmoidalBijector
    perm_indices: jnp.ndarray

    def __init__(self, dim: int, hidden_sizes: List[int], num_components: int, *, key):
        ckey, pkey = jr.split(key)
        
        # Instantiate the networks
        out_features_per_dim = 3 * num_components
        self.conditioner = AutoregressiveConditioner(
            dim=dim, 
            hidden_sizes=hidden_sizes, 
            out_features_per_dim=out_features_per_dim, 
            key=ckey
        )
        self.bijector = DeepSigmoidalBijector(num_components=num_components)
        
        # Generate a fixed random permutation for this specific layer
        self.perm_indices = jr.permutation(pkey, jnp.arange(dim))

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # 1. Generate the flow parameters autoregressively
        params = self.conditioner(x)
        
        # 2. Apply the deep sigmoidal transformation
        z, log_det = self.bijector(x, params)
        
        # 3. Apply the fixed random permutation to both the variables and the log-det
        z_permuted = z[self.perm_indices]
        log_det_permuted = log_det[self.perm_indices]
        
        return z_permuted, log_det_permuted


class InverseAutoregressiveFlow(eqx.Module):
    """
    The complete universal posterior approximator.
    Transforms standard normal noise into complex posterior samples.
    """
    layers: List[FlowLayer]
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, depth: int, hidden_sizes: List[int], num_components: int = 16, *, key):
        self.dim = dim
        keys = jr.split(key, depth)
        
        # Chain multiple flow layers together
        self.layers = [
            FlowLayer(dim, hidden_sizes, num_components, key=k) 
            for k in keys
        ]

    def sample_and_log_prob(self, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Draws a single sample from the base distribution and pushes it through the flow.
        Returns the transformed sample and its exact variational log-density q(z).
        """
        # Base distribution: z_0 ~ N(0, I)
        z = jr.normal(key, (self.dim,))
        
        # Base log-probability: log q(z_0)
        # log N(z | 0, I) = -0.5 * D * log(2pi) - 0.5 * sum(z^2)
        log_q = -0.5 * self.dim * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(z ** 2)
        
        # Push through the autoregressive layers
        for layer in self.layers:
            z, log_det = layer(z)
            # Accumulate the change in volume: q(z_k) = q(z_0) - sum(log_det)
            # We subtract because log_det is calculated in the forward direction
            log_q = log_q - jnp.sum(log_det)
            
        return z, log_q
    
class PyMCTranspiler:
    """
    Bridges PyMC models to JAX for variational flow optimization.
    Extracts the unconstrained continuous log-probability graph and builds 
    flat-to-tree bijection mappings.
    """
    def __init__(self, pymc_model: pm.Model):
        # 1. Validation: Flows require differentiable spaces
        if pymc_model.discrete_value_vars:
            raise ValueError(
                "Inverse Autoregressive Flows require a fully continuous, "
                "differentiable parameter space. Discrete variables detected."
            )
            
        self.cont_vars = pymc_model.continuous_value_vars
        self.var_names = [v.name for v in self.cont_vars]
        
        # 2. Extract the JAX compiled log-probability function
        self.raw_logp_fn = get_jaxified_logp(pymc_model)
        
        # 3. Build shape mappings from the initial point
        init_point = pymc_model.initial_point()
        self.base_tree = {
            v.name: jnp.array(init_point[v.name], dtype=jnp.float64) 
            for v in self.cont_vars
        }
        self.cont_sizes = [self.base_tree[name].size for name in self.var_names]
        self.total_dim = sum(self.cont_sizes)

    def unpack_to_tree(self, flat_array: jnp.ndarray) -> dict:
        """Transforms a flat [D] array from the flow back into the PyMC variable dictionary."""
        new_tree = dict(self.base_tree)
        curr = 0
        for name, size in zip(self.var_names, self.cont_sizes):
            flat_slice = flat_array[curr:curr + size]
            new_tree[name] = flat_slice.reshape(self.base_tree[name].shape)
            curr += size
        return new_tree

    def logp_flat(self, flat_array: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the PyMC log-probability from a flat JAX array."""
        tree_dict = self.unpack_to_tree(flat_array)
        
        # PyMC expects the arguments in the exact order of continuous_value_vars
        logp_val = self.raw_logp_fn([tree_dict[name] for name in self.var_names])
        return jnp.array(logp_val, dtype=jnp.float64)    
    
@eqx.filter_value_and_grad
def compute_loss(flow: InverseAutoregressiveFlow, transpiler: PyMCTranspiler, key: jax.Array, num_samples: int, iwae_k: int = 1) -> jnp.ndarray:
    """
    Computes the Evidence Lower Bound (ELBO) or the Importance Weighted Autoencoder (IWAE) loss.
    """
    batch_keys = jax.random.split(key, num_samples)
    z_samples, log_q = jax.vmap(flow.sample_and_log_prob)(batch_keys)
    log_p = jax.vmap(transpiler.logp_flat)(z_samples)
    
    # Calculate the raw log importance weights
    log_w = log_p - log_q
    
    if iwae_k > 1:
        # --- IWAE Objective (Mass-Covering) ---
        # Group samples into clusters of size k
        num_batches = num_samples // iwae_k
        log_w_grouped = log_w.reshape(num_batches, iwae_k)
        
        # IWAE Loss = -mean( log( 1/k * sum(exp(w)) ) )
        # Using logsumexp for extreme numerical stability
        log_iwae = jax.scipy.special.logsumexp(log_w_grouped, axis=-1) - jnp.log(iwae_k)
        loss = -jnp.mean(log_iwae)
    else:
        # --- Standard ELBO (Mode-Seeking) ---
        loss = -jnp.mean(log_w)
    
    return loss

class IAFOptimizer:
    """
    Universal Variational Posterior Approximator for PyMC.
    Orchestrates Equinox flow training, hardware sharding, and ArviZ inference export.
    """
    def __init__(self, pymc_model: pm.Model, depth: int = 4, hidden_sizes: List[int] = [64, 64], num_components: int = 16, seed: int = 42):
        self.key = jr.PRNGKey(seed)
        
        # 1. Initialize Transpiler
        self.transpiler = PyMCTranspiler(pymc_model)
        
        # 2. Initialize Flow Architecture
        flow_key, self.key = jr.split(self.key)
        self.flow = InverseAutoregressiveFlow(
            dim=self.transpiler.total_dim,
            depth=depth,
            hidden_sizes=hidden_sizes,
            num_components=num_components,
            key=flow_key
        )
        
        # Track hardware state
        self.num_devices = jax.local_device_count()
        print(f"Initialized IAFOptimizer across {self.num_devices} device(s). Parameter space: {self.transpiler.total_dim} dimensions.")

    def fit(self, learning_rate: float = 0.005, num_steps: int = 1000, batch_size: int = 256, weight_decay: float = 1e-4, iwae_k: int = 1, callback=None):
        """
        Maximizes the ELBO or IWAE to train the autoregressive flow using distributed computing.
        """
        # Ensure batch size divides evenly across available physical hardware
        if batch_size % self.num_devices != 0:
            batch_size = (batch_size // self.num_devices) * self.num_devices
        batch_per_device = batch_size // self.num_devices
        
        # Ensure batch_per_device is cleanly divisible by iwae_k for clustering
        if batch_per_device % iwae_k != 0:
            raise ValueError(f"Hardware batch size ({batch_per_device}) must be divisible by iwae_k ({iwae_k}).")
        
        # --- LEARNING RATE SCHEDULER & WEIGHT DECAY ---
        # Dedicate the first 10% of training steps to warming up safely
        warmup_steps = max(1, int(num_steps * 0.10))
        
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate * 0.01,  # Start at 1% of the target learning rate
            peak_value=learning_rate,         # The target learning rate Optuna suggests
            warmup_steps=warmup_steps,
            decay_steps=num_steps,
            end_value=learning_rate * 0.01    # Decay back down to 1% by the final step
        )
        
        # --- GRADIENT CLIPPING ---
        # Chain global norm clipping with AdamW to prevent tail-sample gradient explosions
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
        )
        opt_state = optimizer.init(eqx.filter(self.flow, eqx.is_inexact_array))

        # Define the distributed training step
        # We use pmap to shard the batch, calculate local gradients, and synchronize them via pmean
        @eqx.filter_pmap(axis_name="device")
        def pmap_train_step(flow_shard, opt_state_shard, device_key):
            # Compute loss and gradients on this specific device, passing down the iwae_k parameter
            loss_val, grads = compute_loss(flow_shard, self.transpiler, device_key, batch_per_device, iwae_k)
            
            # Average the gradients and the loss across all hardware devices
            grads = jax.lax.pmean(grads, axis_name="device")
            loss_val = jax.lax.pmean(loss_val, axis_name="device")
            
            # Apply the synchronized updates
            updates, new_opt_state = optimizer.update(grads, opt_state_shard, eqx.filter(flow_shard, eqx.is_inexact_array))
            new_flow = eqx.apply_updates(flow_shard, updates)
            
            return new_flow, new_opt_state, loss_val

        # Replicate the initial flow and optimizer state across all devices
        replicated_flow = jax.device_put_replicated(self.flow, jax.local_devices())
        replicated_opt_state = jax.device_put_replicated(opt_state, jax.local_devices())

        # Execute the optimization loop
        import time
        start_time = time.time()
        
        for step in range(num_steps):
            # Generate unique keys for each device
            step_key, self.key = jr.split(self.key)
            device_keys = jr.split(step_key, self.num_devices)
            
            replicated_flow, replicated_opt_state, loss_array = pmap_train_step(
                replicated_flow, replicated_opt_state, device_keys
            )
            
            if step % max(1, (num_steps // 10)) == 0:
                # Pull the synchronized loss back to the host CPU for printing
                current_loss = loss_array[0].item()
                print(f"Step {step:04d} | Negative {'IWAE' if iwae_k > 1 else 'ELBO'} Loss: {current_loss:.4f}")
                
                # --- OPTUNA INTEGRATION HOOK ---
                if callback is not None:
                    # Pass the step and loss to Optuna. 
                    # If Optuna says prune, we break the loop.
                    if callback(step, current_loss):
                        print(f"Trial pruned at step {step}.")
                        break
        
        # Save the final loss as an attribute so Optuna can retrieve it
        self.final_loss = loss_array[0].item()

        # Collapse the replicated model back to the primary device
        self.flow = tree_map(lambda x: x[0], replicated_flow)
        
        elapsed = time.time() - start_time
        print(f"Optimization complete in {elapsed:.2f} seconds.")

    def sample(self, num_samples: int = 4000) -> az.InferenceData:
        """
        Draws unconstrained samples, computes joint importance weights for VI diagnostics,
        and packages them into an ArviZ InferenceData object.
        """
        sample_key, self.key = jr.split(self.key)
        
        # Compile the sampling and log-probability evaluation
        @eqx.filter_jit
        def generate_samples_and_weights(k):
            keys = jr.split(k, num_samples)
            # 1. Draw from the flow and get the variational log-density (log_q)
            z, log_q = jax.vmap(self.flow.sample_and_log_prob)(keys)
            
            # 2. Evaluate the target PyMC log-probability (log_p)
            log_p = jax.vmap(self.transpiler.logp_flat)(z)
            
            return z, log_q, log_p
            
        flat_samples, log_q, log_p = generate_samples_and_weights(sample_key)
        
        # 3. Calculate Joint Log Importance Weights: log(w) = log(p) - log(q)
        log_weights = np.array(log_p - log_q)
        
        # 4. Translate the flat samples back into the PyMC dictionary structure
        posterior_dict = {}
        for i in range(num_samples):
            tree_draw = self.transpiler.unpack_to_tree(flat_samples[i])
            for var_name, value in tree_draw.items():
                if var_name not in posterior_dict:
                    posterior_dict[var_name] = []
                posterior_dict[var_name].append(value)
                
        # Format for ArviZ: (chains=1, draws=num_samples, *shape)
        for var_name in posterior_dict.keys():
            arr = np.array(posterior_dict[var_name])
            posterior_dict[var_name] = np.expand_dims(arr, axis=0)
            
        # 5. Store the diagnostic weights in the sample_stats group
        sample_stats = {
            "log_weights": np.expand_dims(log_weights, axis=0),
            "log_q": np.expand_dims(np.array(log_q), axis=0),
            "log_p": np.expand_dims(np.array(log_p), axis=0)
        }
            
        return az.from_dict(posterior=posterior_dict, sample_stats=sample_stats)
    
    def summary(self, idata: az.InferenceData) -> pd.DataFrame:
        """
        Generates a custom diagnostic summary specifically for Variational Inference.
        Calculates the global Pareto k statistic, IS-ESS, and standard parameter bounds.
        """
        # Extract the flattened log weights from the InferenceData
        log_weights = idata.sample_stats["log_weights"].values.flatten()
        num_draws = len(log_weights)
        
        # 1. Use ArviZ's PSIS to smooth the weights and extract the Pareto k shape parameter
        smoothed_log_weights, pareto_k = psislw(log_weights)
        
        # 2. Calculate Importance Sampled ESS using the smoothed weights
        weights = np.exp(smoothed_log_weights - np.max(smoothed_log_weights))
        is_ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
        
        # 3. Print the Global Diagnostic Header
        print("=" * 65)
        print(" VARIATIONAL INFERENCE GLOBAL DIAGNOSTICS")
        print("=" * 65)
        print(f" Pareto k statistic : {pareto_k:.4f}")
        
        if pareto_k > 0.7:
            print("  -> WARNING: k > 0.7. The flow missed significant probability mass.")
            print("              Inference on tail events may be biased.")
        elif pareto_k > 0.5:
            print("  -> NOTE: 0.5 < k <= 0.7. Minor tail underestimation.")
            print("           IS-ESS is reliable for parameter recovery.")
        else:
            print("  -> SUCCESS: k <= 0.5. Excellent posterior coverage.")
            
        print(f" IS-ESS             : {is_ess:.0f} (out of {num_draws} total draws)")
        print("=" * 65 + "\n")
        
        # 4. Build and clean the parameter-level DataFrame
        # We request "stats" to get Mean, SD, HDI, and MCSE
        summary_df = az.summary(idata, kind="stats", round_to=3)
        
        # Drop the irrelevant MCMC-specific columns if ArviZ tries to calculate them
        columns_to_drop = ['r_hat', 'ess_bulk', 'ess_tail']
        summary_df = summary_df.drop(columns=[col for col in columns_to_drop if col in summary_df.columns])
        
        return summary_df
    
    def save(self, path: str):
        """Saves the trained flow parameters to a file."""
        eqx.tree_serialise_leaves(path, self.flow)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Loads the flow parameters from a file."""
        # This assumes self.flow is already initialized with the same architecture
        self.flow = eqx.tree_deserialise_leaves(path, self.flow)
        print(f"Model loaded from {path}")