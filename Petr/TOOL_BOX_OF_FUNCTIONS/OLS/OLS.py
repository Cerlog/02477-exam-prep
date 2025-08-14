import jax.numpy as jnp

def get_w_mle(Phi, y, method: str = "solve"):
    """
    Maximum Likelihood Estimate (MLE) of linear regression weights.

    Computes:
        w_MLE = argmin_w ||y - Phi w||^2
              = (Phi^T Phi)^{-1} Phi^T y   (if 'solve')
              = pinv(Phi) y                 (if 'pinv')

    Args:
        Phi : array, shape (N, D)
            Design matrix.
        y   : array, shape (N,) or (N, 1)
            Targets.
        method : {"solve", "pinv"}, default "solve"
            - "solve": uses jnp.linalg.solve on normal equations (fast, needs full rank).
            - "pinv" : uses Moore-Penrose pseudoinverse (robust to rank deficiency).

    Returns:
        w_mle : array, shape (D, 1)
            MLE estimate of weights as a column vector.
    """
    Phi = jnp.asarray(Phi)
    y = jnp.asarray(y).reshape(-1, 1)  # ensure column vector

    if method == "pinv":
        w_mle = jnp.linalg.pinv(Phi) @ y
    else:
        # Normal equations: (Phi^T Phi) w = Phi^T y
        XtX = Phi.T @ Phi
        Xty = Phi.T @ y
        w_mle = jnp.linalg.solve(XtX, Xty)

    return w_mle


def get_sigma2_mle(Phi, y, w_mle=None, ddof: int = 0, method: str = "solve"):
    """
    MLE (or unbiased) estimate of noise variance σ^2 from residuals.

    Computes:
        σ^2 = (1 / (N - ddof)) * ||y - Phi w_MLE||^2
    - ddof=0  -> MLE (division by N)
    - ddof=D  -> unbiased estimator (division by N - D), if you want it

    Args:
        Phi   : array, shape (N, D)
            Design matrix.
        y     : array, shape (N,) or (N, 1)
            Targets.
        w_mle : array, shape (D, 1), optional
            If not provided, computed via get_w_mle(Phi, y, method=method).
        ddof  : int, default 0
            Degrees of freedom to subtract from N in the denominator.
        method: {"solve", "pinv"}, default "solve"
            Passed to get_w_mle if w_mle is None.

    Returns:
        sigma2 : scalar (0-dim array)
            Estimated noise variance.
    """
    Phi = jnp.asarray(Phi)
    y = jnp.asarray(y).reshape(-1, 1)

    if w_mle is None:
        w_mle = get_w_mle(Phi, y, method=method)

    residuals = y - Phi @ w_mle  # (N, 1)
    N = Phi.shape[0]
    denom = N - ddof
    sigma2 = (residuals.T @ residuals).squeeze() / denom  # scalar
    return sigma2
