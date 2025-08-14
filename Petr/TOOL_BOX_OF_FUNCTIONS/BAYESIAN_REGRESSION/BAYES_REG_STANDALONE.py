# blr_functions.py
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, random
from scipy.optimize import minimize


# ---------- helpers ----------

def design_matrix(x):
    """
    Build Φ = [1, x] for 1D inputs.
    x: (N,) or (N,1)
    returns Φ: (N, 2)
    """
    x = jnp.asarray(x).reshape(-1)
    return jnp.column_stack((jnp.ones(x.shape[0]), x))

def _ensure_column(y):
    y = jnp.asarray(y)
    return y.reshape(-1, 1)

def _posterior_core(Phi, y, alpha, beta):
    """
    Core posterior pieces shared by multiple functions.
    Returns:
      A: (D, D)  where A = α I + β ΦᵀΦ
      S: (D, D)  posterior covariance = A^{-1}
      m: (D, 1)  posterior mean = β S Φᵀ y
    """
    Phi = jnp.asarray(Phi)
    y = _ensure_column(y)
    alpha = jnp.asarray(alpha)
    beta  = jnp.asarray(beta)

    D = Phi.shape[1]
    A = alpha * jnp.eye(D) + beta * (Phi.T @ Phi)
    # S = A^{-1} via solve for stability
    S = jnp.linalg.solve(A, jnp.eye(D))
    m = beta * S @ Phi.T @ y
    return A, S, m


# ---------- posterior over weights p(w | y, Φ, α, β) ----------

def posterior_w(Phi, y, alpha, beta):
    """
    Returns (m, S) for p(w|...) = N(m, S).
    """
    _, S, m = _posterior_core(Phi, y, alpha, beta)
    return m, S

def posterior_w_from_xy(x, y, alpha, beta):
    """
    Convenience wrapper using x (1D) -> Φ = [1, x].
    """
    Phi = design_matrix(x)
    return posterior_w(Phi, y, alpha, beta)


# ---------- predictive distributions ----------

def predict_f(Phi_star, Phi_train, y, alpha, beta):
    """
    Predicts latent function f* | y ~ N(μ_f, Σ_f) at rows of Φ*.

    Returns:
      mu_f:  (N*,)
      var_f: (N*,)
    """
    _, S, m = _posterior_core(Phi_train, y, alpha, beta)
    Phi_star = jnp.asarray(Phi_star)
    mu_f = (Phi_star @ m).ravel()
    var_f = jnp.diag(Phi_star @ S @ Phi_star.T)
    return mu_f, var_f

def predict_f_from_xy(x_train, y, x_star, alpha, beta):
    Phi_tr = design_matrix(x_train)
    Phi_st = design_matrix(x_star)
    return predict_f(Phi_st, Phi_tr, y, alpha, beta)

def predict_y(Phi_star, Phi_train, y, alpha, beta):
    """
    Predicts observed outputs y* = f* + ε with ε ~ N(0, β^{-1} I).
    Returns:
      mu_y:  (N*,)
      var_y: (N*,)
    """
    mu_f, var_f = predict_f(Phi_star, Phi_train, y, alpha, beta)
    var_y = var_f + 1.0 / beta
    return mu_f, var_y

def predict_y_from_xy(x_train, y, x_star, alpha, beta):
    Phi_tr = design_matrix(x_train)
    Phi_st = design_matrix(x_star)
    return predict_y(Phi_st, Phi_tr, y, alpha, beta)


# ---------- log marginal likelihood log p(y | α, β) ----------

def log_marginal_likelihood(Phi, y, alpha, beta):
    """
    Bishop PRML (3.86): 
      log p(y|α,β) = (D/2)log α + (N/2)log β - E(m) - (1/2)log|A| - (N/2)log(2π)
    where
      A = α I + β ΦᵀΦ,
      m = β A^{-1} Φᵀ y,
      E(m) = (β/2)||y - Φ m||² + (α/2)||m||².
    """
    Phi = jnp.asarray(Phi)
    y = _ensure_column(y)
    N, D = Phi.shape

    A, S, m = _posterior_core(Phi, y, alpha, beta)
    resid = y - Phi @ m
    Em = (beta / 2.0) * jnp.sum(resid ** 2) + (alpha / 2.0) * jnp.sum(m ** 2)
    sign, logdetA = jnp.linalg.slogdet(A)
    # sign should be +1 for SPD A; we trust that α>0, β>0, ΦᵀΦ PSD
    return (D / 2.0) * jnp.log(alpha) + (N / 2.0) * jnp.log(beta) - Em - 0.5 * logdetA - (N / 2.0) * jnp.log(2.0 * jnp.pi)

def log_marginal_likelihood_xy(x, y, alpha, beta):
    Phi = design_matrix(x)
    return log_marginal_likelihood(Phi, y, alpha, beta)


# ---------- hyperparameter optimization via evidence maximization ----------

def optimize_hyperparameters_xy(x, y, alpha0=1.0, beta0=1.0, method="L-BFGS-B"):
    """
    Maximize log p(y | α, β) over α, β (both constrained > 0) by optimizing θ = (log α, log β).

    Returns:
      alpha_opt, beta_opt, scipy_result
    """
    x = jnp.asarray(x)
    y = _ensure_column(y)

    def nml(theta):
        # theta = [log α, log β]
        alpha = jnp.exp(theta[0])
        beta  = jnp.exp(theta[1])
        return -log_marginal_likelihood_xy(x, y, alpha, beta)

    nml_val_grad = value_and_grad(nml)
    theta0 = np.array([np.log(alpha0), np.log(beta0)], dtype=float)

    def fun_np(th_np):
        th = jnp.asarray(th_np)
        val, grad = nml_val_grad(th)
        return float(val), np.array(grad)

    res = minimize(lambda th: fun_np(th)[0],
                   theta0,
                   jac=lambda th: fun_np(th)[1],
                   method=method)

    alpha_opt = float(np.exp(res.x[0]))
    beta_opt  = float(np.exp(res.x[1]))
    return alpha_opt, beta_opt, res


# ---------- MLEs ----------

def w_mle(Phi, y):
    """
    w_MLE = (ΦᵀΦ)^{-1} Φᵀ y
    Returns (D,1)
    """
    Phi = jnp.asarray(Phi)
    y = _ensure_column(y)
    PhiTPhi_inv = jnp.linalg.inv(Phi.T @ Phi)
    return PhiTPhi_inv @ Phi.T @ y

def w_mle_from_xy(x, y):
    Phi = design_matrix(x)
    return w_mle(Phi, y)

def sigma2_mle(Phi, y):
    """
    σ²_MLE = (1/N) || y - Φ w_MLE ||²
    """
    Phi = jnp.asarray(Phi)
    y = _ensure_column(y)
    w = w_mle(Phi, y)
    resid = y - Phi @ w
    return jnp.sum(resid ** 2) / Phi.shape[0]

def sigma2_mle_from_xy(x, y):
    Phi = design_matrix(x)
    return sigma2_mle(Phi, y)


# ---------- MAP (posterior mode) ----------

def w_map(Phi, y, alpha, beta):
    """
    For Gaussian prior + Gaussian likelihood, MAP = posterior mean m.
    """
    m, _ = posterior_w(Phi, y, alpha, beta)
    return m

def w_map_from_xy(x, y, alpha, beta):
    Phi = design_matrix(x)
    return w_map(Phi, y, alpha, beta)


# ---------- sampling ----------

def sample_prior_w(key, D, alpha, num_samples=1):
    """
    w ~ N(0, α^{-1} I_D)
    Returns array shape (num_samples, D)
    """
    mean = jnp.zeros(D)
    cov  = (1.0 / alpha) * jnp.eye(D)
    return random.multivariate_normal(key, mean, cov, shape=(num_samples,))

def sample_posterior_w(key, Phi, y, alpha, beta, num_samples=1):
    """
    w | y ~ N(m, S)
    Returns array shape (num_samples, D)
    """
    m, S = posterior_w(Phi, y, alpha, beta)
    return random.multivariate_normal(key, m.ravel(), S, shape=(num_samples,))

def sample_posterior_w_from_xy(key, x, y, alpha, beta, num_samples=1):
    Phi = design_matrix(x)
    return sample_posterior_w(key, Phi, y, alpha, beta, num_samples=num_samples)



import jax.numpy as jnp
from jax import random
from blr_functions import *

# synthetic linear data: y = 1.5 + 2.0 x + ε,  ε ~ N(0, 0.1^2)
key = random.PRNGKey(0)
N = 30
x = jnp.linspace(-2, 2, N)
true_w = jnp.array([1.5, 2.0])
sigma = 0.1
y = (true_w[0] + true_w[1] * x + sigma * random.normal(key, (N,))).reshape(-1,1)

# set priors (precisions)
alpha = 1.0     # weight precision
beta  = 1.0 / (sigma ** 2)  # noise precision

# 1) posterior over weights
m, S = posterior_w_from_xy(x, y, alpha, beta)
print("posterior mean (w_map):", m.ravel())
print("posterior cov diag:", jnp.diag(S))

# 2) predictive for f* and y*
x_star = jnp.linspace(-3, 3, 50)
mu_f, var_f = predict_f_from_xy(x, y, x_star, alpha, beta)
mu_y, var_y = predict_y_from_xy(x, y, x_star, alpha, beta)
print("mu_y shape:", mu_y.shape, "var_y shape:", var_y.shape)

# 3) log marginal likelihood
logZ = log_marginal_likelihood_xy(x, y, alpha, beta)
print("log evidence:", float(logZ))

# 4) hyperparameter optimization (evidence maximization)
alpha_opt, beta_opt, res = optimize_hyperparameters_xy(x, y, alpha0=1.0, beta0=10.0)
print("alpha_opt, beta_opt =", alpha_opt, beta_opt)

# 5) MLE estimates
Phi = design_matrix(x)
w_mle_est = w_mle(Phi, y)
sig2_mle  = sigma2_mle(Phi, y)
print("w_MLE:", w_mle_est.ravel(), "sigma2_MLE:", float(sig2_mle))

# 6) MAP (equals posterior mean for Gaussians)
w_map_est = w_map_from_xy(x, y, alpha, beta)
print("w_MAP:", w_map_est.ravel())

# 7) sampling
k1, k2 = random.split(key)
w_prior_samples = sample_prior_w(k1, D=2, alpha=alpha, num_samples=5)
w_post_samples  = sample_posterior_w_from_xy(k2, x, y, alpha, beta, num_samples=5)
print("prior samples shape:", w_prior_samples.shape)
print("posterior samples shape:", w_post_samples.shape)
