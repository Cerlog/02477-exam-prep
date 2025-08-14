import jax.numpy as jnp

def gaussian_credibility_interval(mean, var, level=0.95):
    """
    Compute a symmetric credibility interval for a Gaussian.

    Parameters
    ----------
    mean : float or array
        Posterior mean(s)
    var : float or array
        Posterior variance(s) (σ²)
    level : float
        Credibility level, default 0.95 (95% interval)

    Returns
    -------
    lower : float or array
        Lower bound(s) of the interval
    upper : float or array
        Upper bound(s) of the interval
    """
    std = jnp.sqrt(var)
    z = 1.96 if jnp.isclose(level, 0.95) else jnp.abs(jax.scipy.stats.norm.ppf((1 - level) / 2))
    lower = mean - z * std
    upper = mean + z * std
    return lower, upper


# usage 
#lower, upper = gaussian_credibility_interval(mean_plug_in, sigma_hat**2, level=0.95)
#print(lower, upper)