"""
This script trains a minimal policy gradient agent on the "Toy environment", which is the
defined in Figure 2 of the GFlowNet Foundations paper, Bengio et al (JMLR, 2023):

.. _a link: https://jmlr.org/papers/v24/22-0364.html
"""

"""
Key differences with rl.py:
- JAX instead of PyTorch
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

import optax
import equinox as eqx

from tqdm import tqdm
import time

### COMMON VARIABLES ###

float_type = jnp.float32
key = random.PRNGKey(0)
do_print = True

### ENVIRONMENT ###

discount_factor = 1.0 # No discounting
batch_size = 1

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

    def __call__(self, x):
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

# Stateless, pure loss function (preferred in JAX) #! Good loss fn example!
# def loss_fn(model, x, y, weight=1.0):
#     pred = model(x)
#     return weight * jnp.mean((pred - y) ** 2)

# @eqx.filter_jit
# def make_step(model, opt_state, x, y, optimizer): #! Good make_step fn example!
#     loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, 0.5)
#     updates, opt_state = optimizer.update(grads, opt_state, params=model)
#     model = eqx.apply_updates(model, updates)
#     return model, opt_state, loss

def policy_gradient_loss(model):
    pass
    
    
### GRAPH MASKS ###

mask_dict = {}
for state in range(n_states):
    mask_invalid = [False if s in connections_dict[state] else True for s in range(n_states)]
    mask_invalid += [-1 not in connections_dict[state]]
    mask_dict[state] = mask_invalid
    
### TRAIN ###

if not do_print:
    pbar = tqdm(
        initial=0,
        total=n_train_steps // batch_size,
    )

start_time = time.time()

