"""
Bayes Toolkit (JAX)
===================
A compact, readable collection of Bayesian building blocks implemented in JAX:
- Distributions (log-pdfs, sampling helpers) for common priors & likelihoods
- Conjugate model utilities (updates + posterior predictive)
- Variational Inference (CAVI-style GMM + Black-box VI base)
- Sampling methods (ancestral, rejection, importance, MH, simple Gibbs pattern)

Designed for study & small experiments — no external deps beyond JAX.

Usage: import this file and call functions/classes. Each section includes
short examples in comments.

Conventions
-----------
- All arrays are jnp.ndarrays. Randomness uses jax.random.PRNGKey.
- Shapes are documented per function. Batch-friendly where easy.
- Gamma uses (shape=a, rate=b) parameterization unless stated.
- Wishart uses scale matrix W and dof nu (S ~ Wishart(W, nu)).

Tested with: jax >= 0.4
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, Optional

import jax
import jax.numpy as jnp
from jax import random, jit, grad, value_and_grad
from jax.scipy.special import gammaln, logsumexp

Array = jnp.ndarray
Key = jax.Array

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def symmetrize(M: Array) -> Array:
    return 0.5 * (M + M.T)


def safe_cholesky(S: Array, jitter: float = 1e-6) -> Array:
    """Cholesky with jitter for numerical stability."""
    S = symmetrize(S)
    eye = jnp.eye(S.shape[-1])
    def _chol(A):
        return jnp.linalg.cholesky(A + jitter * eye)
    return jax.lax.cond(
        jnp.all(jnp.isfinite(jnp.linalg.eigvalsh(S))) & (jnp.min(jnp.linalg.eigvalsh(S)) > 0),
        lambda _: jnp.linalg.cholesky(S + jitter * eye),
        lambda _: _chol(S),
        operand=None,
    )


# -----------------------------------------------------------------------------
# Basic distributions: logpdfs & sampling helpers
# -----------------------------------------------------------------------------
# NOTE: For clarity we implement simple versions. Extend as needed.

# Univariate Normal N(x | m, v) with variance v
@jit
def log_norm1(x: Array, m: Array, v: Array) -> Array:
    return -0.5 * (jnp.log(2 * jnp.pi * v) + (x - m) ** 2 / v)


# Bernoulli (x in {0,1}) with prob p
@jit
def log_bernoulli(x: Array, p: Array) -> Array:
    p = jnp.clip(p, 1e-12, 1 - 1e-12)
    return x * jnp.log(p) + (1 - x) * jnp.log1p(-p)


# Binomial (y successes out of n) with prob p
@jit
def log_binomial(y: Array, n: Array, p: Array) -> Array:
    p = jnp.clip(p, 1e-12, 1 - 1e-12)
    log_binom_coeff = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
    return log_binom_coeff + y * jnp.log(p) + (n - y) * jnp.log1p(-p)


# Beta(a, b) prior (a>0, b>0)
@jit
def log_beta(theta: Array, a: Array, b: Array) -> Array:
    theta = jnp.clip(theta, 1e-12, 1 - 1e-12)
    logB = gammaln(a) + gammaln(b) - gammaln(a + b)
    return (a - 1) * jnp.log(theta) + (b - 1) * jnp.log1p(-theta) - logB


# Gamma(a, b) with shape a, rate b (>0)
@jit
def log_gamma_pdf(x: Array, a: Array, b: Array) -> Array:
    x = jnp.clip(x, 1e-30, jnp.inf)
    return a * jnp.log(b) - gammaln(a) + (a - 1) * jnp.log(x) - b * x


# Categorical (one draw) with probs pi, input is index k in [0..K-1]
@jit
def log_categorical(k: Array, pi: Array) -> Array:
    pi = pi / pi.sum(-1, keepdims=True)
    return jnp.log(jnp.take_along_axis(pi, k[..., None], axis=-1)[..., 0] + 1e-30)


# Dirichlet(alpha) over probability vectors pi
@jit
def log_dirichlet(pi: Array, alpha: Array) -> Array:
    pi = jnp.clip(pi, 1e-30, jnp.inf)
    logC = gammaln(alpha.sum(-1)) - jnp.sum(gammaln(alpha), axis=-1)
    return logC + jnp.sum((alpha - 1) * jnp.log(pi), axis=-1)


# Multivariate Normal N(x | m, S) with full covariance S
@jit
def log_mvn(x: Array, m: Array, S: Array) -> Array:
    d = x.shape[-1]
    L = safe_cholesky(S)
    y = jax.scipy.linalg.solve_triangular(L, (x - m)[..., None], lower=True, trans='N')
    quad = jnp.sum(y[..., 0] ** 2, axis=-1)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return -0.5 * (d * jnp.log(2 * jnp.pi) + logdet + quad)


# Wishart(W, nu): precision-like random matrix with df nu and scale W (PD)
# Log-pdf (up to constants): see standard definition
@jit
def log_wishart(S: Array, W: Array, nu: float) -> Array:
    """Log pdf of Wishart(S | W, nu).
    Uses scale W so E[S] = nu * W (for nu > d-1), d = dim.
    """
    d = S.shape[-1]
    Lw = safe_cholesky(W)
    Ls = safe_cholesky(S)
    logdetW = 2 * jnp.sum(jnp.log(jnp.diag(Lw)))
    logdetS = 2 * jnp.sum(jnp.log(jnp.diag(Ls)))
    # Multivariate gamma log Γ_d(nu/2)
    def log_multigamma(a, p):
        return (p * (p - 1) * jnp.log(jnp.pi) / 4.0
                + jnp.sum(gammaln(a - (jnp.arange(p) / 2.0))))
    term1 = (nu - d - 1) / 2.0 * logdetS
    term2 = -0.5 * jnp.trace(jnp.linalg.solve(W, S))
    term3 = - (nu * d / 2.0) * jnp.log(2.0) - (nu / 2.0) * logdetW - log_multigamma(nu / 2.0, d)
    return term1 + term2 + term3


# Sampling helpers -------------------------------------------------------------

# Univariate Normal: x ~ N(m, v)
# Usage:
#   key = random.PRNGKey(0)
#   x = sample_norm1(key, m=0.0, v=1.0)
@jit
def sample_norm1(key: Key, m: Array, v: Array) -> Array:
    return m + jnp.sqrt(v) * random.normal(key, shape=jnp.shape(m))

# Multivariate Normal with full covariance (already above): sample_mvn()
# Add a convenience for diagonal covariance: S = diag(v)
# Usage:
#   key = random.PRNGKey(0)
#   m = jnp.zeros(3); v = jnp.array([1., 2., 0.5])
#   x = sample_mvn_diag(key, m, v)
@jit
def sample_mvn_diag(key: Key, m: Array, v: Array) -> Array:
    z = random.normal(key, shape=m.shape)
    return m + jnp.sqrt(v) * z


def sample_beta(key: Key, a: Array, b: Array) -> Array:
    u1, u2 = random.gamma(key, a), random.gamma(random.split(key, 2)[1], b)
    return u1 / (u1 + u2)


def sample_dirichlet(key: Key, alpha: Array) -> Array:
    g = random.gamma(key, alpha)
    return g / g.sum(-1, keepdims=True)


def sample_categorical(key: Key, pi: Array) -> Array:
    return random.categorical(key, jnp.log(pi), axis=-1)


def sample_mvn(key: Key, m: Array, S: Array) -> Array:
    L = safe_cholesky(S)
    z = random.normal(key, shape=m.shape)
    return m + L @ z


def sample_wishart(key: Key, W: Array, nu: float) -> Array:
    """Bartlett decomposition sampler for Wishart(W, nu)."""
    d = W.shape[-1]
    keys = random.split(key, d + 1)
    A = jnp.zeros((d, d))
    def body(i, A):
        chi2 = random.gamma(keys[i], (nu - i) / 2.0) * 2.0
        A = A.at[i, i].set(jnp.sqrt(chi2))
        off = random.normal(keys[(i + 1) % (d + 1)], (d - i - 1,))
        A = A.at[i + 1:, i].set(off)
        return A
    A = jax.lax.fori_loop(0, d, body, A)
    Lw = safe_cholesky(W)
    C = Lw @ A
    return C @ C.T


# -----------------------------------------------------------------------------
# Conjugate model utilities
# -----------------------------------------------------------------------------
# 1) Beta-Binomial
# -----------------------------------------------------------------------------
@dataclass
class BetaBinomial:
    a0: float
    b0: float

    def posterior(self, y: int, N: int) -> Tuple[float, float]:
        """Return posterior Beta params (a, b)."""
        return self.a0 + y, self.b0 + (N - y)

    def post_mean(self, y: int, N: int) -> float:
        a, b = self.posterior(y, N)
        return a / (a + b)

    def log_evidence(self, y: int, N: int) -> float:
        # p(y|N) under Beta-Binomial
        a, b = self.a0, self.b0
        return (gammaln(N + 1) - gammaln(y + 1) - gammaln(N - y + 1)
                + gammaln(a + y) + gammaln(b + N - y) - gammaln(a + b + N)
                - (gammaln(a) + gammaln(b) - gammaln(a + b)))

    def pred_bernoulli_mean(self, y: int, N: int) -> float:
        """P(next success) = a/(a+b)."""
        return self.post_mean(y, N)

# Example:
# bb = BetaBinomial(a0=1., b0=1.)
# a,b = bb.posterior(y=2, N=5)  # (3,4)
# p_next = bb.pred_bernoulli_mean(y=2, N=5)


# 2) Normal-Gamma model for unknown mean and precision
#    x_i ~ N(mu, 1/tau), mu|tau ~ N(mu0, 1/(lambda0*tau)), tau ~ Gamma(a0, b0)
# -----------------------------------------------------------------------------
@dataclass
class NormalGamma:
    mu0: float
    lambda0: float
    a0: float
    b0: float

    def posterior(self, x: Array) -> Tuple[float, float, float, float]:
        N = x.shape[0]
        xbar = jnp.mean(x)
        # Update
        lambda_n = self.lambda0 + N
        mu_n = (self.lambda0 * self.mu0 + N * xbar) / lambda_n
        a_n = self.a0 + 0.5 * (N + 1)
        # sum of squares around xbar
        ss = jnp.sum((x - xbar) ** 2)
        b_n = self.b0 + 0.5 * ss + 0.5 * (self.lambda0 * N / lambda_n) * (xbar - self.mu0) ** 2
        return mu_n, lambda_n, a_n, b_n

    def post_mean_var_mu(self, x: Array) -> Tuple[float, float]:
        mu_n, lambda_n, a_n, b_n = self.posterior(x)
        # E[tau] = a_n / b_n
        v_mu = 1.0 / ((a_n / b_n) * lambda_n)
        return mu_n, v_mu

    def pred_student_t_params(self, x: Array) -> Tuple[float, float, float]:
        """Return parameters of posterior predictive for a single new x*:
        x* ~ StudentT(df=2*a_n, loc=mu_n, scale^2 = b_n/(a_n*lambda_n))
        """
        mu_n, lambda_n, a_n, b_n = self.posterior(x)
        df = 2.0 * a_n
        scale2 = b_n / (a_n * lambda_n)
        return df, mu_n, scale2

# Example:
# ng = NormalGamma(mu0=0., lambda0=1., a0=1., b0=1.)
# df, loc, s2 = ng.pred_student_t_params(jnp.array([1.0, 0.7, 1.2]))


# 3) Dirichlet-Multinomial (a.k.a. Dirichlet-Categorical)
# -----------------------------------------------------------------------------
@dataclass
class DirichletCategorical:
    alpha0: Array  # shape (K,)

    def posterior(self, counts: Array) -> Array:
        return self.alpha0 + counts

    def mean(self, counts: Array) -> Array:
        alpha = self.posterior(counts)
        return alpha / alpha.sum()

    def log_evidence(self, counts: Array) -> float:
        alpha0 = self.alpha0
        return (gammaln(alpha0.sum()) - jnp.sum(gammaln(alpha0))
                + jnp.sum(gammaln(alpha0 + counts)) - gammaln(alpha0.sum() + counts.sum()))

# Example:
# dc = DirichletCategorical(alpha0=jnp.ones(3))
# alpha = dc.posterior(jnp.array([5,3,2]))
# pi_mean = alpha/alpha.sum()


# -----------------------------------------------------------------------------
# Bayesian GMM with NIW priors — compact VI (CAVI-style)
# -----------------------------------------------------------------------------
@dataclass
class NIWPrior:
    m0: Array      # (D,)
    beta0: float
    W0: Array      # (D,D) PD
    nu0: float     # > D-1
    alpha0: float  # symmetric Dirichlet concentration per component


@dataclass
class VariationalGMM:
    D: int
    K: int
    prior: NIWPrior
    max_itt: int = 500
    tol: float = 1e-6
    seed: int = 0

    # Learned variational params
    # mixture
    alpha: Optional[Array] = None  # (K,)
    # component posteriors
    m: Optional[Array] = None      # (K,D)
    beta: Optional[Array] = None   # (K,)
    W: Optional[Array] = None      # (K,D,D)
    nu: Optional[Array] = None     # (K,)

    # Cached responsibilities
    r: Optional[Array] = None      # (N,K)

    def _init_params(self):
        D, K = self.D, self.K
        pr = self.prior
        self.alpha = jnp.ones(K) * pr.alpha0
        self.m = jnp.tile(pr.m0, (K, 1))
        self.beta = jnp.ones(K) * pr.beta0
        self.W = jnp.tile(pr.W0, (K, 1, 1))
        self.nu = jnp.ones(K) * pr.nu0

    def _expected_log_pi(self) -> Array:
        # E[log pi_k] under Dir(alpha) = psi(alpha_k) - psi(sum alpha)
        psi = jax.scipy.special.digamma
        return psi(self.alpha) - psi(jnp.sum(self.alpha))

    def _expected_log_lambda(self) -> Array:
        # E[log |Lambda_k|] for Wishart(W_k, nu_k)
        d = self.D
        psi = jax.scipy.special.digamma
        term = jnp.sum(psi(0.5 * (self.nu - jnp.arange(d))), axis=-1)
        logdetW = jnp.linalg.slogdet(self.W)[1]
        return term + d * jnp.log(2.0) + logdetW

    def _mahalanobis(self, X: Array) -> Array:
        # returns (N,K) expected quadratic forms E[(x - m_k)^T Lambda_k (x - m_k)]
        N = X.shape[0]
        Xm = X[:, None, :] - self.m[None, :, :]           # (N,K,D)
        quad = jnp.einsum('nkd,kde,nke->nk', Xm, self.nu[:, None, None] * self.W, Xm)
        return quad + self.D / self.beta  # E[(x-m)^T Lambda (x-m)] + trace term

    def _e_step(self, X: Array) -> Array:
        N, D, K = X.shape[0], self.D, self.K
        ElogPi = self._expected_log_pi()                  # (K,)
        ElogLambda = self._expected_log_lambda()          # (K,)
        quad = self._mahalanobis(X)                       # (N,K)
        log_rho = (ElogPi + 0.5 * ElogLambda - 0.5 * (D * jnp.log(2 * jnp.pi) + quad))
        log_r = log_rho - logsumexp(log_rho, axis=1, keepdims=True)
        r = jnp.exp(log_r)
        return r

    def _m_step(self, X: Array, r: Array):
        pr = self.prior
        Nk = r.sum(axis=0) + 1e-12                        # (K,)
        xbar_k = (r.T @ X) / Nk[:, None]                   # (K,D)
        # scatter matrices
        Xm = X[:, None, :] - xbar_k[None, :, :]
        Sk = jnp.einsum('nk,nkd,nke->kde', r, Xm, Xm) / Nk[:, None, None]  # (K,D,D)

        # Update q(pi)
        self.alpha = pr.alpha0 + Nk
        # Update q(mu, Lambda)
        self.beta = pr.beta0 + Nk
        self.m = (pr.beta0 * pr.m0 + Nk[:, None] * xbar_k) / self.beta[:, None]
        diff = xbar_k - pr.m0
        W_inv = jnp.linalg.inv(pr.W0)[None, :, :] + Nk[:, None, None] * Sk + (pr.beta0 * Nk)[:, None, None] / self.beta[:, None, None] * jnp.einsum('ki,kj->kij', diff, diff)
        self.W = jnp.linalg.inv(W_inv)
        self.nu = pr.nu0 + Nk

    def fit(self, X: Array) -> "VariationalGMM":
        self._init_params()
        elbo_prev = -jnp.inf
        for _ in range(self.max_itt):
            r = self._e_step(X)
            self._m_step(X, r)
            self.r = r
            # simple convergence check on responsibilities
            elbo = jnp.sum(jnp.log(jnp.maximum(r.sum(axis=1), 1e-30)))  # proxy
            if jnp.abs(elbo - elbo_prev) < self.tol:
                break
            elbo_prev = elbo
        return self

    def component_probs(self, X: Array) -> Array:
        return self._e_step(X)

    def posterior_mixture_mean(self) -> Array:
        return self.alpha / jnp.sum(self.alpha)

# Example:
# D, K = 2, 3
# prior = NIWPrior(m0=jnp.zeros(D), beta0=1., W0=jnp.eye(D), nu0=D+2, alpha0=1.)
# X = random.normal(random.PRNGKey(0), (300, D))
# vi = VariationalGMM(D, K, prior, max_itt=200).fit(X)
# r = vi.component_probs(X)


# -----------------------------------------------------------------------------
# Black-box VI: mean-field Gaussian with reparameterization + Adam
# -----------------------------------------------------------------------------
@dataclass
class Adam:
    lr: float = 1e-2
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    def init(self, theta: Array):
        m = jnp.zeros_like(theta)
        v = jnp.zeros_like(theta)
        t = jnp.array(0)
        return (theta, m, v, t)

    def step(self, state, g):
        theta, m, v, t = state
        t = t + 1
        m = self.b1 * m + (1 - self.b1) * g
        v = self.b2 * v + (1 - self.b2) * (g * g)
        mhat = m / (1 - self.b1 ** t)
        vhat = v / (1 - self.b2 ** t)
        theta = theta - self.lr * mhat / (jnp.sqrt(vhat) + self.eps)
        return (theta, m, v, t)


@dataclass
class BBVI:
    log_joint_fn: Callable[[Array], float]  # returns log p(y, w)
    D: int
    key: Key
    lr: float = 1e-2
    steps: int = 2000
    S: int = 8  # MC samples per step

    def fit(self, init_m: Optional[Array] = None, init_logv: Optional[Array] = None) -> Tuple[Array, Array]:
        key = self.key
        m = jnp.zeros((self.D,)) if init_m is None else init_m
        logv = jnp.zeros((self.D,)) if init_logv is None else init_logv
        adam = Adam(self.lr)
        state = adam.init(jnp.concatenate([m, logv]))

        def elbo_from_params(theta: Array, key: Key) -> float:
            m, logv = theta[: self.D], theta[self.D :]
            v = jnp.exp(logv)
            std = jnp.sqrt(v)
            keys = random.split(key, self.S)
            def one(key_s):
                eps = random.normal(key_s, (self.D,))
                w = m + std * eps
                return self.log_joint_fn(w) - (0.5 * jnp.log(2 * jnp.pi * v) + (eps ** 2) / 2.0).sum()
            samples = jax.vmap(one)(keys)
            return jnp.mean(samples)

        val_and_grad_elbo = jax.value_and_grad(lambda th, k: -elbo_from_params(th, k))

        for _ in range(self.steps):
            key, sub = random.split(key)
            loss, g = val_and_grad_elbo(state[0], sub)
            state = adam.step(state, g)
        theta = state[0]
        m, logv = theta[: self.D], theta[self.D :]
        return m, jnp.exp(logv)

# Example (logistic regression skeleton):
# def make_log_joint(X, y, sigma_w=10.0):
#     def log_joint(w):
#         z = X @ w
#         ll = jnp.sum(y * jax.nn.log_sigmoid(z) + (1 - y) * jax.nn.log_sigmoid(-z))
#         lp = -0.5 * jnp.sum(jnp.log(2*jnp.pi*sigma_w**2) + (w**2)/(sigma_w**2))
#         return ll + lp
#     return log_joint
# key = random.PRNGKey(0)
# X = random.normal(key, (200, 5)); y = random.bernoulli(key, 0.5, (200,)).astype(jnp.float32)
# log_joint = make_log_joint(X, y)
# bbvi = BBVI(log_joint, D=5, key=key, steps=1000, lr=5e-2)
# m, v = bbvi.fit()


# -----------------------------------------------------------------------------
# Sampling methods: ancestral, rejection, importance, MH, simple Gibbs pattern
# -----------------------------------------------------------------------------

# 1) Ancestral sampling demo utilities ----------------------------------------
# For Bayesian networks you’d sample parents -> children. Here we include a
# small helper for GMM generative sampling.

def sample_gmm(key: Key, pi: Array, mus: Array, covs: Array, N: int) -> Tuple[Array, Array]:
    """Sample N points from a GMM.
    pi: (K,), mus: (K,D), covs: (K,D,D)
    Returns X: (N,D), z: (N,)
    """
    K = pi.shape[0]
    keys = random.split(key, N + 1)
    z = random.categorical(keys[0], jnp.log(pi), shape=(N,))
    def one(i, _):
        k = z[i]
        x = sample_mvn(keys[i + 1], mus[k], covs[k])
        return _, x
    _, X = jax.lax.scan(one, None, jnp.arange(N))
    return X, z


# 2) Rejection sampling --------------------------------------------------------

def rejection_sample(key: Key, logp: Callable[[Array], float], 
                     logg: Callable[[Array], float],
                     sample_g: Callable[[Key], Array],
                     logM: float,
                     max_trials: int = 10_000) -> Tuple[Array, int]:
    """One draw from p using envelope g with bound p(x) <= M g(x).
    Returns (x, trials_used)."""
    def body_fun(carry):
        key, i = carry
        key, kx, ku = random.split(key, 3)
        x = sample_g(kx)
        u = random.uniform(ku)
        accept = jnp.log(u) < (logp(x) - logg(x) - logM)
        return (key, i + 1), (x, accept)

    def cond_fun(state):
        (key, i), (x, accept) = state
        return (~accept) & (i < max_trials)

    init_state = ((key, 0), (jnp.array(0.0), False))
    def step(state):
        (key, i), _ = state
        (key, i), (x, accept) = body_fun((key, i))
        return (key, i), (x, accept)

    state = init_state
    for _ in range(max_trials):
        state = step(state)
        if not cond_fun(state):
            break
    (_, trials), (x, accept) = state
    return x, int(trials)


# 3) Importance sampling -------------------------------------------------------

def importance_estimate(key: Key, logp: Callable[[Array], float],
                        logg: Callable[[Array], float],
                        sample_g: Callable[[Key, Tuple[int]], Array],
                        h: Callable[[Array], Array],
                        S: int = 10_000) -> Tuple[float, float]:
    """Estimate E_p[h(X)] using samples from g.
    Returns (estimate, effective_sample_size).
    sample_g(key, (S,)) -> samples shape (S, D?)
    """
    key_s = random.split(key, S)
    X = sample_g(key, (S,))
    lw = jax.vmap(lambda x: logp(x) - logg(x))(X)
    w = jnp.exp(lw - jnp.max(lw))  # stabilize
    w = w / w.sum()
    est = jnp.sum(w * jax.vmap(h)(X))
    ess = 1.0 / jnp.sum(w ** 2)
    return float(est), float(ess)


# 4) Metropolis–Hastings (random-walk Gaussian) -------------------------------

def metropolis_hastings(key: Key, logpost: Callable[[Array], float],
                        init: Array, steps: int = 5_000, 
                        prop_scale: float = 0.1) -> Tuple[Array, float]:
    """Return samples (steps, D) and acceptance rate."""
    D = init.shape[0]
    keys = random.split(key, steps * 2)
    x = init
    samples = []
    accepts = 0
    for t in range(steps):
        eps = random.normal(keys[2 * t], (D,)) * prop_scale
        x_prop = x + eps
        loga = logpost(x_prop) - logpost(x)
        u = jnp.log(random.uniform(keys[2 * t + 1]))
        accept = u < loga
        x = jax.lax.select(accept, x_prop, x)
        samples.append(x)
        accepts += accept.astype(jnp.int32)
    return jnp.stack(samples), float(accepts / steps)


# 5) Gibbs pattern -------------------------------------------------------------
# You provide conditional samplers for blocks; we iterate them.

def gibbs(key: Key, inits: Dict[str, Array], 
          conditionals: Dict[str, Callable[[Key, Dict[str, Array]], Array]],
          steps: int = 1000) -> Dict[str, Array]:
    """Generic Gibbs: conditionals[var](key, state)-> new sample.
    Returns dict of traces with shape (steps, ...)
    """
    state = {k: v for k, v in inits.items()}
    traces = {k: [] for k in inits}
    keys = random.split(key, steps * len(inits))
    tkey = 0
    for t in range(steps):
        for name, sampler in conditionals.items():
            state[name] = sampler(keys[tkey], state)
            tkey += 1
        for k in inits:
            traces[k].append(state[k])
    return {k: jnp.stack(v) for k, v in traces.items()}


# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
