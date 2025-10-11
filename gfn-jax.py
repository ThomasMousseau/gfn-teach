"""
This script trains a minimal policy gradient agent on the "Toy environment", which is the
defined in Figure 2 of the GFlowNet Foundations paper, Bengio et al (JMLR, 2023):

.. _a link: https://jmlr.org/papers/v24/22-0364.html
"""

"""
Key differences with gfn.py:
- JAX and Equinox instead of PyTorch (could have used pure JAX but Equinox makes the syntax closer to PyTorch plus it handles basic things like parameter initialization and model updates)
- JAX is hardware agnostic (CPU/GPU/TPU), thus we don't need to specify device
- JAX builds its function around a single input then extends to batches using vmap
- Since JAX is built around functional programming, we will avoid classes when possible otherwise we will need to explicitely define its PyTree
- Keys for random number generation are explicitely passed around in initialization and function calls
- Uses Optax and Equinox instead of torch.optim and torch.nn
- Equinox modules uses __call__ method instead of forward method
- Functions using auto-differentiation need to have their first argument as the parameters to differentiate with respect to (usually model parameters)
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jaxtyping import Array, Float, Int


import cProfile
import pstats
from pstats import SortKey


import optax
import equinox as eqx

from tqdm import tqdm
import time

### COMMON VARIABLES ###

float_type = jnp.float32
key = random.PRNGKey(0)
do_print = False

### ENVIRONMENT ###

# A dictionary of connections: the keys of the dictionary are the indices of the
# states, and the values are the indices of the states to which each state is
# connected.
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

# A dictionary of rewards: the keys of the dictionary are the indices of the
# states, and the values are their rewards.
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

# Linear layer following an embedding layer. The inputs are the state indices and the
# outputs are tensors with dimensionality equal to the number of actions: the number of
# states plus one, for the end-of-sequence (EOS) action.

class Policy(eqx.Module):
    embedding: eqx.nn.Embedding #! Review PyTree good practices
    linear: eqx.nn.Linear #! Review PyTree good practices
    
    def __init__(self, n_states, key):
        super().__init__()
        key1, key2 = random.split(key) # Keys for parameter initialization
        self.embedding = eqx.nn.Embedding(n_states, n_states, key=key1)
        self.linear = eqx.nn.Linear(
            n_states,
            n_states + 1,
            dtype=float_type,
            key=key2,
        )

    def __call__(self, x: int) -> Float[Array, "n_actions"]:
        x = self.embedding(x)
        x = self.linear(x)
        return x
    
policy = Policy(n_states, key)

### OPTIMIZER ###

n_train_steps = 2000
learning_rate = 0.01
momentum = 0.9
optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)
optimizer_state = optimizer.init(eqx.filter(policy, eqx.is_array)) # Tells optimizer which parameters to optimize knowing that they are arrays

### LOSS FUNCTION ###

def model_flow_matching_loss(
    model: Policy, 
    loginflow: Float[Array, ""], # Scalar from jax.numpy
    state: int, 
    children: Int[Array, "n_states"], # 1D array
    n_valid_children: int
) -> Float[Array, ""]:
    valid_children = children[:n_valid_children]  
    # outflow_logits = jax.vmap(model)(state)[children]
    outflow_logits = model(state)[valid_children] 
    logoutflow = jax.nn.logsumexp(outflow_logits)
    return jnp.square(logoutflow - loginflow)
    
# Decorator to Just In Time (JIT) compile the function for speed
@eqx.filter_jit
def make_step(model, opt_state, optimizer,  loginflow, state, children, n_children): 
    loss, grads = eqx.filter_value_and_grad(model_flow_matching_loss)(model, loginflow, state, children, n_children) # Computes the loss and its gradients w.r.t. model parameters
    updates, opt_state = optimizer.update(grads, opt_state, params=model) # Does the auto-diff and computes the updates
    model = eqx.apply_updates(model, updates) # Equivalent of optimizer.step() in PyTorch
    return model, opt_state, loss


def compute_trajectory_loss_and_update(
    model, opt_state, optimizer,
    loginflows: Float[Array, "n_steps"],
    logoutflows: Float[Array, "n_steps"],
    n_steps: int
):
    # Compute flow matching loss for entire trajectory
    losses = jnp.square(logoutflows - loginflows)
    loss = jnp.sum(losses) / n_steps
    
    # Compute gradients and update
    grads = eqx.filter_grad(lambda m: loss)(model)
    updates, opt_state = optimizer.update(grads, opt_state, params=model)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, loss
    
### GRAPH MASKS ###

mask_dict = {}
for state in range(n_states):
    mask_invalid = [False if s in connections_dict[state] else True for s in range(n_states)]
    mask_invalid += [-1 not in connections_dict[state]]
    mask_dict[state] = mask_invalid
    
### TRAIN ###
def train(key: random.PRNGKey, policy: Policy, optimizer: optax.GradientTransformation, optimizer_state: optax.OptState):
    if not do_print:
        pbar = tqdm(
            initial=0,
            total=n_train_steps,
        )

    key, *subkeys = random.split(key, n_train_steps * n_states) 
    key_idx = 0

    for step in range(n_train_steps): #! Should this be a jax.lax.scan?
        
        # Initialize a trajectory with state 0 and trajectory not done
        state = 0
        traj_done = False
        n_steps = 0

        # Initialize loss to zero
        loss = 0.0

        if do_print:
            print(f"\nIteration {step}")
            print(f"\tTrajectory 0 -> ", end="")

        # Sample actions until trajectory is done
        while not traj_done:
            
            # Use pre-generated key
            subkey = subkeys[key_idx]
            key_idx += 1

            # Build the mask of invalid actions from the current state
            mask_invalid = mask_dict[state]

            # Obtain policy log-flows from the current state, mask invalid actions and
            # sample action
            logits_sampled = policy(state)
            logits_masked = jnp.where(jnp.array(mask_invalid), -jnp.inf, logits_sampled) # Mask invalid actions
            action = int(random.categorical(subkey, logits_masked))
            n_steps += 1

            # Update state, flag of done trajectory and get reward
            if action == n_states:
                traj_done = True
                reward = rewards_dict[state]
                if do_print:
                    print(f"EOS (reward {reward})")
            else:
                state = action
                reward = 0
                if do_print:
                    print(f"{action} -> ", end="")
                    
            # Obtain in-flows:
            # - Get parents of state
            # - Obtain log-flows from each parent to state
            # - Take the log of the sum of the exponential log-flows
            if traj_done:
                parents = [state]
            else:
                parents = [s for s in range(n_states) if state in connections_dict[s]]

            #inflows_logits = jax.vmap(policy)(parents)[:, action]
            inflows_logits = jnp.array([policy(parent)[action] for parent in parents])
            loginflow = jax.nn.logsumexp(inflows_logits)
            
            # Obtain out-flows:
            # - Obtain children of state
            # - Obtain log-flows from the state and mask out transitions that are not
            # children
            # - Take the log of the sum of the exponential log-flows
            # - If the trajectory is done, the log-outflow is just the log-reward
            # Prepare padded children (ALWAYS, even when done)
            if traj_done:
                children_padded = jnp.zeros(n_states, dtype=jnp.int32)
                n_children = 0
                logoutflow = jnp.log(float(reward))
            else:
                children_list = connections_dict[state]
                n_children = len(children_list)
                children_padded = jnp.pad(
                    jnp.array(children_list, dtype=jnp.int32),
                    (0, n_states - n_children), #(before, after)
                    mode='constant',
                    constant_values=0
                )
                outflow_logits = policy(state)[children_padded[:n_children]]
                logoutflow = jax.nn.logsumexp(outflow_logits)

            # Accumulate loss inline (simpler than calling make_step)
            loss = loss + jnp.square(logoutflow - loginflow)
        
        loss /= n_steps
        
        # Use make_step ONLY for gradient update with the last step's data
        grads = eqx.filter_grad(lambda m: loss)(policy)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params=policy)
        policy = eqx.apply_updates(policy, updates)
        
        if do_print:
            print(f"\tTotal loss: {loss:.4f}")
        if not do_print:
            pbar.update(1)
            pbar.set_description(f"Loss: {loss:.4f}")
        
### EVALUATE ###

def eval(key: random.PRNGKey, policy: Policy):
    n_samples = 2000

    # A dictionary to count the number of times each terminal state is sampled
    samples_dict = {
        3: 0,
        4: 0,
        6: 0,
        8: 0,
        9: 0,
        10: 0,
    }

    for step in range(n_samples):
        
        # Initialize a trajectory with state 0 and trajectory not done
        state = 0
        traj_done = False
        
        # Sample actions until trajectory is done
        while not traj_done:
            
            # Build the mask of invalid actions from the current state
            mask_invalid = jnp.array(mask_dict[state])
            
            # Obtain policy log-flows from the current state, mask invalid actions and
            # sample action
            key, subkey = random.split(key)
            logits_sampled = vmap(policy)(jnp.array([state]))
            logits_sampled = jnp.where(mask_invalid, -jnp.inf, logits_sampled)
            action = random.categorical(subkey, logits_sampled)
            
            # Update state, flag of done trajectory and get reward
            if action == n_states:
                traj_done = True
                samples_dict[state] += 1
            else:
                state = int(action[0])

    # Print results
    print("\nEvaluation: \n")
    z = sum(rewards_dict.values())
    absolute_error = 0.0
    for sample, count in samples_dict.items():
        p_sampled = count / n_samples
        p_true = rewards_dict[sample] / z
        absolute_error += abs(p_sampled - p_true)
        print(
            "- Sample {:2d} was generated with probability {:.2f} and the "
            "actual probability is {:.2f}".format(sample, p_sampled, p_true)
        )
    mae = absolute_error / len(samples_dict)
    print("Mean absolute error: {:.2f}".format(mae))

if __name__ == "__main__":
    # start = time.perf_counter(); train(key, policy, optimizer, optimizer_state); print(f"Training: {time.perf_counter() - start:.2f}s")
    # start = time.perf_counter(); eval(key, policy); print(f"Evaluation: {time.perf_counter() - start:.2f}s")
    
    #! Wouldn't recommend using profiler.trace 
    # jax.profiler.start_trace("/tmp/jax-trace", create_perfetto_link=True)
    # train(key, policy, optimizer, optimizer_state)
    # jax.profiler.stop_trace()
    # print("Trace saved. Open the Perfetto link printed above.")
    
    # Profile training
    profiler = cProfile.Profile()
    profiler.enable()
    train(key, policy, optimizer, optimizer_state)
    profiler.disable()
    
    # Print results sorted by cumulative time
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Show top 20 functions
    
    
    


    

