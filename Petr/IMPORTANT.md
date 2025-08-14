import jax
import jax.numpy as jnp
from jax import random
import numpy as np
log_npdf = lambda x, m, v: -(x-m)**2/(2*v) - 0.5*jnp.log(2*jnp.pi*v)

print(f"Log joint at w_map {log_joint(w_map_5):.2f}")   !!!! :.2f


import jax.numpy as jnp
from jax.scipy.stats import binom  # Note: you'll need this import for binom_dist.logpmf

sigmoid = lambda x: 1./(1 + jnp.exp(-x))

log_npdf = lambda x, m, v: -(x-m)**2/(2*v) - 0.5*jnp.log(2*jnp.pi*v)