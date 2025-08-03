import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from jax import random
from scipy.optimize import minimize
import pylab as plt
from scipy.stats import multivariate_normal as mvn
def metropolis(log_target, num_params, tau, num_iter, theta_init=None, seed=0):
    """
    Run the Metropolis-Hastings MCMC algorithm to draw samples from a target distribution.

    Parameters
    ----------
    log_target : callable
        Function that evaluates the log of the unnormalized target distribution.
        Input shape: (num_params,) → Output shape: () (scalar)
        Should return log(p(theta)) up to a constant.

    num_params : int
        Number of parameters (dimensionality of theta ∈ R^D).

    tau : float
        Standard deviation of the Gaussian proposal distribution.
        Proposal: theta_star = theta_cur + tau * epsilon, where epsilon ~ N(0, I).
        Must be > 0.

    num_iter : int
        Number of sampling steps (excluding the initial sample).

    theta_init : jnp.ndarray of shape (num_params,), optional
        Starting point of the Markov chain. If None, initializes at zero.

    seed : int
        Seed for JAX's random number generator to ensure reproducibility.

    Returns
    -------
    thetas : jnp.ndarray, shape = (num_iter + 1, num_params)
        All MCMC samples including the initial point. Each row is a sample.

    Notes
    -----
    The algorithm follows the Metropolis-Hastings logic:
      1. Sample proposal: theta_star = theta_cur + tau * epsilon
      2. Compute acceptance ratio: r = p(theta_star) / p(theta_cur)
         (done in log domain as log_r = log_p_star - log_p_cur)
      3. Accept with probability A = min(1, exp(log_r))
         - If accepted: move to theta_star
         - If rejected: stay at theta_cur

    Shapes Summary
    --------------
    - theta_init          : (num_params,)
    - theta_cur           : (num_params,)
    - theta_star          : (num_params,)
    - epsilon             : (num_params,)
    - thetas              : (num_iter + 1, num_params)
    - accepts             : (num_iter,) (list of integers: 0 or 1)

    Examples
    --------

    # 1. Sample from 1D Gaussian N(1, 3)
    >>> def log_npdf(x, mu, var):
    ...     return -0.5 * jnp.log(2 * jnp.pi * var) - 0.5 * ((x - mu) ** 2) / var
    >>> log_target = lambda x: log_npdf(x, 1.0, 3.0)
    >>> samples = metropolis(log_target, num_params=1, tau=2.0, num_iter=10000)
    >>> print(jnp.mean(samples), jnp.var(samples))  # Should be ~1 and ~3

    # 2. Sample from 2D standard Gaussian N([0, 0], I)
    >>> def log_target_2d(x):
    ...     return -0.5 * jnp.sum(x**2) - jnp.log(2 * jnp.pi)
    >>> samples_2d = metropolis(log_target_2d, num_params=2, tau=1.0, num_iter=10000)

    # 3. Sample from 10D Gaussian with diagonal covariance
    >>> mu = jnp.linspace(0, 9, 10)
    >>> var = jnp.ones(10) * 2.0
    >>> def log_nd_10(x):
    ...     return -0.5 * jnp.sum((x - mu)**2 / var) - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * var))
    >>> samples_10d = metropolis(log_nd_10, num_params=10, tau=1.0, num_iter=5000)

    # 4. Banana-shaped distribution (non-Gaussian)
    >>> def log_banana(x):
    ...     x1, x2 = x[0], x[1]
    ...     return -x1**2 / 200 - 0.5 * (x2 + 0.03 * x1**2 - 3)**2
    >>> samples_banana = metropolis(log_banana, num_params=2, tau=1.5, num_iter=20000)

    # 5. Discrete-like distribution (e.g. log-prob that spikes)
    >>> def log_spiky(x):
    ...     return -10 * jnp.floor(x[0])**2
    >>> samples_spiky = metropolis(log_spiky, num_params=1, tau=0.5, num_iter=10000)

    Tips
    ----
    - Monitor the printed acceptance ratio. Good values are typically between 20% and 40%.
    - Use small `tau` if most proposals are rejected; use larger `tau` if almost all are accepted (may indicate slow exploration).
    - Burn-in and thinning are optional post-processing steps depending on your use case.

    """
    # -------------------------------------------------------
    # !Initialize random number generator
    # -------------------------------------------------------
    key = random.PRNGKey(seed)

    # -------------------------------------------------------
    # !Set starting point (default = 0 vector)
    # -------------------------------------------------------
    if theta_init is None:
        theta_init = jnp.zeros((num_params,))  # shape (num_params,)

    # -------------------------------------------------------
    # !Prepare containers for MCMC samples and acceptances
    # -------------------------------------------------------
    thetas = [theta_init]         # list of samples: each is (num_params,)
    accepts = []                  # list of 0/1 indicators

    # Evaluate log of target at initial point
    log_p_theta = log_target(theta_init)

    for k in range(num_iter):

        # -------------------------------------------------------
        # !Split key into proposal and acceptance subkeys
        # -------------------------------------------------------
        key, key_proposal, key_accept = random.split(key, num=3)

        # -------------------------------------------------------
        # !Sample from proposal distribution:
        #   theta_star = theta_cur + tau * epsilon
        # -------------------------------------------------------
        theta_cur = thetas[-1]                                 # shape (num_params,)
        epsilon = random.normal(key_proposal, shape=(num_params,))  # shape (num_params,)
        theta_star = theta_cur + tau * epsilon                 # shape (num_params,)

        # -------------------------------------------------------
        # !Compute acceptance probability: A = min(1, exp(log_r))
        # -------------------------------------------------------
        log_p_theta_star = log_target(theta_star)              # scalar
        log_r = log_p_theta_star - log_p_theta
        A = jnp.minimum(1.0, jnp.exp(log_r))                   # scalar ∈ [0, 1]

        # -------------------------------------------------------
        # !Accept or reject the proposal
        # -------------------------------------------------------
        u = random.uniform(key_accept)
        if u < A:
            # Accept the proposal
            theta_next = theta_star
            log_p_theta = log_p_theta_star
            accepts.append(1)
        else:
            # Reject the proposal
            theta_next = theta_cur
            accepts.append(0)

        # -------------------------------------------------------
        # !Append the new sample to the chain
        # -------------------------------------------------------
        thetas.append(theta_next)

    # -------------------------------------------------------
    # !Report acceptance rate (useful for diagnostics)
    # -------------------------------------------------------
    acceptance_rate = jnp.mean(jnp.array(accepts))
    print(f"Acceptance ratio: {acceptance_rate:.2f}")

    # -------------------------------------------------------
    # !Convert list to array of shape (num_iter + 1, num_params)
    # -------------------------------------------------------
    print(len(thetas))
    thetas = jnp.stack(thetas)
    print(thetas.shape)
    # -------------------------------------------------------
    # !Shape check for debugging
    # -------------------------------------------------------
    assert thetas.shape == (num_iter + 1, num_params), (
        f"Expected shape {(num_iter + 1, num_params)}, but got {thetas.shape}"
    )

    return thetas