import jax.numpy as jnp

def bayesian_linear_regression_posterior(X, y, alpha, sigma2):
    """
    Computes the posterior mean and covariance of w given X and y.

    Parameters:
    -----------
    X : np.ndarray, shape (N, D)
        Design matrix.
    y : np.ndarray, shape (N,) or (N, 1)
        Output vector.
    alpha : float
        Prior precision (1 / variance of prior on w).
    sigma2 : float
        Noise variance (likelihood variance).

    Returns:
    --------
    mu : np.ndarray, shape (D, 1)
        Posterior mean of w.
    Sigma : np.ndarray, shape (D, D)
        Posterior covariance of w.
    """
    # Ensure y is column vector
    y = y.reshape(-1, 1)  # shape: (N, 1)

    N, D = X.shape
    I = jnp.eye(D)

    XtX = X.T @ X            # shape: (D, D)
    Xty = X.T @ y            # shape: (D, 1)

    # Posterior precision matrix
    A = alpha * I + (1 / sigma2) * XtX  # shape: (D, D)

    # Posterior covariance = inverse of A
    Sigma = jnp.linalg.inv(A)           # shape: (D, D)

    # Posterior mean
    mu = Sigma @ ((1 / sigma2) * Xty)  # shape: (D, 1)

    return mu, Sigma
