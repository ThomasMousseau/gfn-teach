"""
Optimized JAX implementation of GFlowNet training on the Toy environment.
This version uses jax.lax.scan and precomputed graph structures for maximum performance.

Key optimizations:
- Precomputed graph structure as JAX arrays (no Python loops during training)
- jax.lax.scan for both training steps and trajectory sampling
- Fully JIT-compiled functions
- Vectorized operations where possible
- Proper functional programming patterns for JAX
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from jaxtyping import Array, Float, Int
from typing import NamedTuple

import optax
import equinox as eqx

from tqdm import tqdm
import time

### COMMON VARIABLES ###

float_type = jnp.float32
key = random.PRNGKey(0)
do_print = False

### ENVIRONMENT ###

connections_dict = {
    0: (1, 2),
    1: (3,),
    2: (3, 4),
    3: (5, -1),
    4: (6, -1),
    5: (7, 8),
    6: (8, 10, -1),
    7: (9,),
    8: (9, -1),
    9: (-1,),
    10: (-1,),
}

rewards_dict = {
    3: 30,
    4: 14,
    6: 23,
    8: 10,
    9: 30,
    10: 5,
}

n_states = len(connections_dict)

### POLICY MODEL ###

class Policy(eqx.Module):
    embedding: eqx.nn.Embedding
    linear: eqx.nn.Linear
    
    def __init__(self, n_states, key):
        key1, key2 = random.split(key)
        self.embedding = eqx.nn.Embedding(n_states, n_states, key=key1)
        self.linear = eqx.nn.Linear(
            n_states,
            n_states + 1,
            use_bias=True,
            key=key2,
        )

    def __call__(self, x: Int[Array, ""]) -> Float[Array, "n_actions"]:
        x = self.embedding(x)
        x = self.linear(x)
        return x

### PRECOMPUTE GRAPH STRUCTURE ###

# Compute mask dictionary first (needed for evaluation)
mask_dict = {}
for state in range(n_states):
    mask_invalid = [False if s in connections_dict[state] else True for s in range(n_states)]
    mask_invalid += [-1 not in connections_dict[state]]
    mask_dict[state] = jnp.array(mask_invalid)

# Convert mask_dict to a 2D array for vectorized operations
mask_array = jnp.array([mask_dict[i] for i in range(n_states)])

# Precompute parent information for each (state, action) pair
max_parents = 0
parent_counts = {}
parent_lists = {}

for child_state in range(n_states):
    for action_idx in range(n_states + 1):
        if action_idx == n_states:  # EOS action
            # For EOS, the only parent is the state itself
            parents = [child_state]
        else:
            # Find all states that can transition to action_idx
            parents = [s for s in range(n_states) if action_idx in connections_dict.get(s, [])]
        
        parent_lists[(child_state, action_idx)] = parents
        parent_counts[(child_state, action_idx)] = len(parents)
        max_parents = max(max_parents, len(parents))

# Create padded parent arrays
parent_indices = jnp.full((n_states, n_states + 1, max_parents), 0, dtype=jnp.int32)
n_parents = jnp.zeros((n_states, n_states + 1), dtype=jnp.int32)

for (child_state, action_idx), parents in parent_lists.items():
    n_parents = n_parents.at[child_state, action_idx].set(len(parents))
    for i, parent in enumerate(parents):
        parent_indices = parent_indices.at[child_state, action_idx, i].set(parent)

# Precompute children structure (excluding EOS)
max_children = max(len([c for c in children if c != -1]) for children in connections_dict.values())
children_indices = jnp.full((n_states, max_children), 0, dtype=jnp.int32)
n_children_array = jnp.zeros(n_states, dtype=jnp.int32)

for state, children in connections_dict.items():
    valid_children = [c for c in children if c != -1]
    n_children_array = n_children_array.at[state].set(len(valid_children))
    for i, child in enumerate(valid_children):
        children_indices = children_indices.at[state, i].set(child)

# Convert rewards to array
rewards_array = jnp.zeros(n_states, dtype=float_type)
for state, reward in rewards_dict.items():
    rewards_array = rewards_array.at[state].set(float(reward))

### OPTIMIZER ###

n_train_steps = 2000
learning_rate = 0.01
momentum = 0.9
optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)

### TRAJECTORY SAMPLING ###

class TrajStepCarry(NamedTuple):
    state: Int[Array, ""]
    done: Int[Array, ""]
    step_count: Int[Array, ""]
    total_loss: Float[Array, ""]
    key: jax.Array

def make_trajectory_step(policy_model):
    """Create a trajectory step function that closes over the policy."""
    def trajectory_step(carry: TrajStepCarry, _) -> tuple[TrajStepCarry, None]:
        """Single step in trajectory sampling."""
        state, done, step_count, total_loss, key = carry
        key, subkey = random.split(key)
        
        # Get logits and sample action
        mask_invalid = mask_array[state]
        logits = policy_model(state)
        logits_masked = jnp.where(mask_invalid, -jnp.inf, logits)
        action = random.categorical(subkey, logits_masked)
        
        # Determine if this is an EOS action
        is_eos = action == n_states
        
        # Next state (stays same if EOS)
        next_state = jnp.where(is_eos, state, action)
        
        # Get reward
        reward = rewards_array[state]
        
        # Compute inflow (vectorized over parents)
        parents_for_action = parent_indices[next_state, action]
        n_p = n_parents[next_state, action]
        
        # Vectorized computation of parent logits
        def get_parent_logit(parent_idx):
            return policy_model(parent_idx)[action]
        
        parent_logits = vmap(get_parent_logit)(parents_for_action)
        
        # Mask out invalid parents (beyond n_p)
        parent_mask = jnp.arange(max_parents) >= n_p
        parent_logits = jnp.where(parent_mask, -jnp.inf, parent_logits)
        loginflow = jax.nn.logsumexp(parent_logits)
        
        # Compute outflow
        def compute_outflow_non_eos():
            children = children_indices[state]
            n_c = n_children_array[state]
            child_logits = policy_model(state)[children]
            child_mask = jnp.arange(max_children) >= n_c
            child_logits = jnp.where(child_mask, -jnp.inf, child_logits)
            return jax.nn.logsumexp(child_logits)
        
        logoutflow = jnp.where(
            is_eos,
            jnp.log(reward + 1e-10),
            compute_outflow_non_eos()
        )
        
        # Compute step loss (only if not already done)
        step_loss = jnp.square(logoutflow - loginflow)
        total_loss = jnp.where(done, total_loss, total_loss + step_loss)
        step_count = jnp.where(done, step_count, step_count + 1)
        
        # Update done flag
        done = done | is_eos
        
        new_carry = TrajStepCarry(next_state, done, step_count, total_loss, key)
        return new_carry, None
    
    return trajectory_step

def sample_trajectory(policy_model, key, max_steps=50):
    """Sample a complete trajectory and compute its loss."""
    trajectory_step = make_trajectory_step(policy_model)
    
    init_carry = TrajStepCarry(
        state=jnp.array(0, dtype=jnp.int32),
        done=jnp.array(0, dtype=jnp.int32),
        step_count=jnp.array(0, dtype=jnp.int32),
        total_loss=jnp.array(0.0, dtype=float_type),
        key=key
    )
    
    final_carry, _ = jax.lax.scan(trajectory_step, init_carry, None, length=max_steps)
    
    # Average loss over steps
    avg_loss = final_carry.total_loss / jnp.maximum(final_carry.step_count, 1)
    
    return avg_loss

### TRAINING ###

class TrainStepCarry(NamedTuple):
    model: Policy
    opt_state: optax.OptState
    key: jax.Array

@eqx.filter_jit
def train_step(carry: TrainStepCarry, _) -> tuple[TrainStepCarry, Float[Array, ""]]:
    """Single training step: sample trajectory, compute gradients, update model."""
    model, opt_state, key = carry
    key, subkey = random.split(key)
    
    # Compute loss and gradients
    def loss_fn(m):
        return sample_trajectory(m, subkey)
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    
    # Update model
    updates, opt_state = optimizer.update(grads, opt_state, params=model)
    model = eqx.apply_updates(model, updates)
    
    new_carry = TrainStepCarry(model, opt_state, key)
    return new_carry, loss

def train(key: random.PRNGKey, policy: Policy, optimizer: optax.GradientTransformation, optimizer_state: optax.OptState) -> Policy:
    """Train the policy using jax.lax.scan for maximum performance."""
    if not do_print:
        pbar = tqdm(initial=0, total=n_train_steps, desc="Training")
    
    # Initialize carry
    init_carry = TrainStepCarry(policy, optimizer_state, key)
    
    # Run training loop with scan
    final_carry, losses = jax.lax.scan(train_step, init_carry, None, length=n_train_steps)
    
    # Extract final model
    final_policy = final_carry.model
    
    # Update progress bar
    if not do_print:
        # Convert losses to numpy for progress bar
        losses_np = jnp.array(losses)
        for i, loss in enumerate(losses_np):
            pbar.update(1)
            pbar.set_description(f"Loss: {float(loss):.4f}")
        pbar.close()
    else:
        for i, loss in enumerate(losses):
            print(f"Iteration {i}: Loss: {float(loss):.4f}")
    
    return final_policy

### EVALUATION ###

def eval(key: random.PRNGKey, policy: Policy):
    """Evaluate the trained policy by sampling trajectories."""
    n_samples = 2000

    samples_dict = {
        3: 0,
        4: 0,
        6: 0,
        8: 0,
        9: 0,
        10: 0,
    }

    for step in range(n_samples):
        state = 0
        traj_done = False
        
        while not traj_done:
            mask_invalid = mask_array[state]
            
            key, subkey = random.split(key)
            logits_sampled = policy(state)
            logits_masked = jnp.where(mask_invalid, -jnp.inf, logits_sampled)
            action = int(random.categorical(subkey, logits_masked))
            
            if action == n_states:
                traj_done = True
                samples_dict[state] += 1
            else:
                state = action

    # Print results
    print("\nEvaluation:\n")
    z = sum(rewards_dict.values())
    absolute_error = 0.0
    for sample, count in samples_dict.items():
        p_sampled = count / n_samples
        p_true = rewards_dict[sample] / z
        absolute_error += abs(p_sampled - p_true)
        print(
            f"- Sample {sample:2d} was generated with probability {p_sampled:.2f} and the "
            f"actual probability is {p_true:.2f}"
        )
    mae = absolute_error / len(samples_dict)
    print(f"Mean absolute error: {mae:.2f}")

if __name__ == "__main__":
    # Initialize policy
    policy = Policy(n_states, key)
    optimizer_state = optimizer.init(eqx.filter(policy, eqx.is_array))
    
    # Training
    start = time.perf_counter()
    trained_policy = train(key, policy, optimizer, optimizer_state)
    train_time = time.perf_counter() - start
    print(f"\nTraining time: {train_time:.2f}s")
    
    # Evaluation
    start = time.perf_counter()
    eval(key, trained_policy)
    eval_time = time.perf_counter() - start
    print(f"Evaluation time: {eval_time:.2f}s")
