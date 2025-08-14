import jax
import jax.numpy as jnp
from jax import random


def sample_normal_from_variance(mu: float, var: float, shape=(1,), seed: int = 0) -> jnp.ndarray:
    """
    Samples from a normal distribution N(mu, var) using a fixed seed.

    Args:
        mu (float): Mean of the normal distribution.
        var (float): Variance (σ²) of the normal distribution.
        shape (tuple, optional): Shape of the output array. Defaults to (1,).
        seed (int, optional): Seed for PRNG key. Defaults to 0.

    Returns:
        jnp.ndarray: Samples from N(mu, var), shape = `shape`.

    Example:
        >>> w = sample_normal_from_variance(0.0, 1.0, shape=(3,), seed=42)
        >>> print(w)
        [-0.20584244 -1.0984622  -0.7460445 ]
    """
    key = random.PRNGKey(seed)
    sigma = jnp.sqrt(var)
    return mu + sigma * random.normal(key, shape=shape)



import jax
import jax.numpy as jnp
from jax import random

def sample_multivariate_normal(mu: jnp.ndarray, cov: jnp.ndarray, shape=(1,), seed: int = 0) -> jnp.ndarray:
    """
    Samples from a multivariate normal distribution N(mu, cov) using a fixed seed.

    Args:
        mu (jnp.ndarray): Mean vector of shape (D,).
        cov (jnp.ndarray): Covariance matrix of shape (D, D).
        shape (tuple, optional): Number of samples to draw. Defaults to (1,).
                                 E.g., shape=(S,) returns S samples of shape (D,).
        seed (int, optional): Seed for PRNG key. Defaults to 0.

    Returns:
        jnp.ndarray: Samples from N(mu, cov), shape = `shape + mu.shape`

    Example:
        >>> mu = jnp.array([0.0, 1.0])
        >>> cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        >>> sample_multivariate_normal(mu, cov, shape=(3,), seed=42)
        Array of shape (3, 2)
    """
    key = random.PRNGKey(seed)

    # Cholesky decomposition: cov = L @ L.T
    L = jnp.linalg.cholesky(cov)  # shape (D, D)

    # Sample standard normals: shape = (S, D)
    std_samples = random.normal(key, shape=shape + mu.shape)

    # Transform: mu + L @ z
    return mu + std_samples @ L.T
