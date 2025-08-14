import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from jax import random
from scipy.optimize import minimize
import pylab as plt
from scipy.stats import multivariate_normal as mvn



def plot_summary(ax, x, s, interval=95, num_samples=0, sample_color='k', sample_alpha=0.4, interval_alpha=0.25, color='r', legend=True, title="", plot_mean=True, plot_median=False, label="", seed=0):
    
    b = 0.5*(100 - interval)
    
    lower = jnp.percentile(s, b, axis=0).T
    upper = jnp.percentile(s, 100-b, axis=0).T
    
    if plot_median:
        median = jnp.percentile(s, [50], axis=0).T
        lab = 'Median'
        if len(label) > 0:
            lab += " %s" % label
        ax.plot(x.ravel(), median, label=lab, color=color, linewidth=4)
        
    if plot_mean:
        mean = jnp.mean(s, axis=0).T
        lab = 'Mean'
        if len(label) > 0:
            lab += " %s" % label
        ax.plot(x.ravel(), mean, '--', label=lab, color=color, linewidth=4)
    ax.fill_between(x.ravel(), lower.ravel(), upper.ravel(), color=color, alpha=interval_alpha, label='%d%% Interval' % interval)    
    
    if num_samples > 0:
        jnp.random.seed(seed)
        idx_samples = jnp.random.choice(range(len(s)), size=num_samples, replace=False)
        ax.plot(x, s[idx_samples, :].T, color=sample_color, alpha=sample_alpha);
    
    if legend:
        ax.legend(loc='best')
        
    if len(title) > 0:
        ax.set_title(title, fontweight='bold')
        

# class BayesianLinearRegression(object):
    
#     def __init__(self, Phi, y, alpha=1., beta=1.):
        
#         # store data and hyperparameters
#         self.Phi, self.y = Phi, y
#         self.N, self.D = Phi.shape
#         self.alpha, self.beta = alpha, beta
        
#         # compute posterior distribution
#         self.m, self.S = self.compute_posterior(alpha, beta)
#         self.log_marginal_likelihood = self.compute_marginal_likelihood(alpha, beta)

#         # perform sanity check of shapes/dimensions
#         self.check_dimensions()

#     def set_hyperparameters(self, alpha, beta):
#         self.alpha = alpha
#         self.beta = beta
#         self.m, self.S = self.compute_posterior(alpha, beta)

#     def check_dimensions(self):
#         D = self.D
#         assert self.m.shape == (D, 1), f"Wrong shape for posterior mean.\nFor D = {D}, the shape of the posterior mean must be ({D}, 1), but the actual shape is ({self.m.shape})"
#         assert self.S.shape == (D, D), f"Wrong shape for posterior covariance.\nFor D = {D}, the shape of the posterior mean must be ({D}, {D}), , but the actual shape is ({self.S.shape})"
#         # assert self.log_marginal_likelihood.shape == (), f"Wrong shape for log_marginal_likelihood.\nThe shape of must be (), but the actual shape is ({self.log_marginal_likelihood.shape})"

#     def compute_posterior(self, alpha, beta):
#         """ computes the posterior N(w|m, S) and return m, S.
#             Shape of m and S must be (D, 1) and (D, D), respectively  """
        
#         #############################################
#         # Insert your solution here
#         #############################################
        
#         # compute prior and posterior precision 
#         inv_S0 = alpha*jnp.identity(self.D)
#         A = inv_S0 + beta*(self.Phi.T@self.Phi)
        
#         # compute mean and covariance 
#         m = beta*jnp.linalg.solve(A, self.Phi.T)@self.y   # eq. (2) above
#         S = jnp.linalg.inv(A)                             # eq. (1) above
        
#         #############################################
#         # End of solution
#         #############################################
#         return m, S
      
#     def generate_prior_samples(self, num_samples):
#         """ generate samples from the prior  """
#         return multivariate_normal.rvs(jnp.zeros(len(self.m)), (1/self.alpha)*jnp.identity(len(self.m)), size=num_samples)
    
#     def generate_posterior_samples(self, num_samples):
#         """ generate samples from the posterior  """
#         return multivariate_normal.rvs(self.m.ravel(), self.S, size=num_samples)
    
#     def predict_f(self, Phi):
#         """ computes posterior mean (mu_f) and variance (var_f) of f(phi(x)) for each row in Phi-matrix.
#             If Phi is a [N, D]-matrix, then the shapes of both mu_f and var_f must be (N,)
#             The function returns (mu_f, var_f)
#         """
#         mu_f = (Phi@self.m).ravel()   
#         var_f = jnp.diag(Phi@self.S@Phi.T)   
        
#         # check dimensions before returning values
#         assert mu_f.shape == (Phi.shape[0],), "Shape of mu_f seems wrong. Check your implementation"
#         assert var_f.shape == (Phi.shape[0],), "Shape of var_f seems wrong. Check your implementation"
#         return mu_f, var_f
        
#     def predict_y(self, Phi):
#         """ returns posterior predictive mean (mu_y) and variance (var_y) of y = f(phi(x)) + e for each row in Phi-matrix.
#             If Phi is a [N, D]-matrix, then the shapes of both mu_y and var_y must be (N,).
#             The function returns (mu_y, var_y)
#         """
#         mu_f, var_f = self.predict_f(Phi)
#         mu_y = mu_f                  
#         var_y = var_f + 1/self.beta  

#         # check dimensions before returning values
#         assert mu_y.shape == (Phi.shape[0],), "Shape of mu_y seems wrong. Check your implementation"
#         assert var_y.shape == (Phi.shape[0],), "Shape of var_y seems wrong. Check your implementation"
#         return mu_y, var_y
        
    
#     def compute_marginal_likelihood(self, alpha, beta):
#         """ computes and returns log marginal likelihood p(y|alpha, beta) """
#         inv_S0 = alpha*jnp.identity(self.D)
#         A = inv_S0 + beta*(self.Phi.T@self.Phi)
#         m = beta*jnp.linalg.solve(A, self.Phi.T)@self.y   # (eq. 3.53 in Bishop)
#         S = jnp.linalg.inv(A)                             # (eq. 3.54 in Bishop)
#         Em = beta/2*jnp.sum((self.y - self.Phi@m)**2) + alpha/2*jnp.sum(m**2)
#         return self.D/2*jnp.log(alpha) + self.N/2*jnp.log(beta) - Em - 0.5*jnp.linalg.slogdet(A)[1] - self.N/2*jnp.log(2*jnp.pi)
         

#     def optimize_hyperparameters(self):
#         # optimizes hyperparameters using marginal likelihood
#         theta0 = jnp.array((jnp.log(self.alpha), jnp.log(self.beta)))
#         def negative_marginal_likelihood(theta):
#             alpha, beta = jnp.exp(theta[0]), jnp.exp(theta[1])
#             return -self.compute_marginal_likelihood(alpha, beta)

#         result = minimize(value_and_grad(negative_marginal_likelihood), theta0, jac=True)

#         # store new hyperparameters and recompute posterior
#         theta_opt = result.x
#         self.alpha, self.beta = jnp.exp(theta_opt[0]), jnp.exp(theta_opt[1])
#         self.m, self.S = self.compute_posterior(self.alpha, self.beta)
#         self.log_marginal_likelihood = self.compute_marginal_likelihood(self.alpha, self.beta)


def metropolis(log_target, num_params, tau, num_iter, theta_init=None, seed=0):    
    """ Runs a Metropolis-Hastings sampler 
    
        Arguments:
        log_target:         function for evaluating the log target distribution, i.e. log \tilde{p}(theta). The function expect a parameter of size num_params.
        num_params:         number of parameters of the joint distribution (integer)
        tau:                standard deviation of the Gaussian proposal distribution (positive real)
        num_iter:           number of iterations (integer)
        theta_init:         vector of initial parameters (jnp.array with shape (num_params) or None)        
        seed:               seed (integer)

        returns
        thetas              jnp.array with MCMC samples (jnp.array with shape (num_iter+1, num_params))
    """ 
    
    # set initial key
    key = random.PRNGKey(seed)

    if theta_init is None:
        theta_init = jnp.zeros((num_params))
    
    # prepare lists 
    thetas = [theta_init]
    accepts = []
    log_p_theta = log_target(theta_init)
    
    for k in range(num_iter):

        # update keys: key_proposal for sampling proposal distribution and key_accept for deciding whether to accept or reject.
        key, key_proposal, key_accept = random.split(key, num=3)

        # get the last value for theta and generate new proposal candidate
        theta_cur = thetas[-1]
        theta_star = theta_cur + tau*random.normal(key_proposal, shape=(num_params, ))
        
        # evaluate the log density for the candidate sample
        log_p_theta_star = log_target(theta_star)

        # compute acceptance probability
        log_r = log_p_theta_star - log_p_theta
        A = min(1, jnp.exp(log_r))
        
        # accept new candidate with probability A
        if random.uniform(key_accept) < A:
            theta_next = theta_star
            log_p_theta = log_p_theta_star
            accepts.append(1)
        else:
            theta_next = theta_cur
            accepts.append(0)

        thetas.append(theta_next)


        
    print('Acceptance ratio: %3.2f' % jnp.mean(jnp.array(accepts)))
        
    # return as jnp.array
    thetas = jnp.stack(thetas)

    # check dimensions and return
    assert thetas.shape == (num_iter+1, num_params), f'The shape of thetas was expected to be ({num_iter+1}, {num_params}), but the actual shape was {thetas.shape}. Please check your code.'
    return thetas, accepts


# implementation borrow from
# from https://github.com/jwalton3141/jwalton3141.github.io/blob/master/assets/posts/ESS/rwmh.py

def gelman_rubin(x):
    """
    Estimate the (over-)dispersed marginal posterior variance across multiple MCMC chains.

    This implements the variance estimator used in the original Gelman–Rubin diagnostic:
        s^2 = W * (n - 1) / n + B_over_n
    where W is the average within-chain variance and B_over_n is the between-chain variance
    scaled by the number of iterations n.

    Parameters
    ----------
    x : array-like of shape (m_chains, n_iters)
        Samples for one scalar parameter collected from multiple parallel MCMC chains.
        Each row corresponds to a chain; each column is a (post-warmup) iteration.
        Note: This function uses `jax.numpy` (`jnp`) operations; `x` can be a NumPy array
        or a JAX DeviceArray. If it is a NumPy array, it will be implicitly converted.

    Returns
    -------
    s2 : scalar (float)
        Over-dispersed estimate of the marginal posterior variance for the parameter.

    Notes
    -----
    - Assumes all chains have equal length and are approximately from the stationary distribution
      (i.e., warm-up/burn-in has been discarded).
    - `m_chains` must be >= 2, and `n_iters` must be >= 2.
    - This is *not* R-hat; it is the variance estimator used inside the R-hat formula.

    Examples
    --------
    >>> # Suppose you ran 4 chains for 1000 iterations for one parameter:
    >>> # x has shape (4, 1000)
    >>> s2 = gelman_rubin(x)
    >>> float(s2)  # doctest: +SKIP
    """
    m_chains, n_iters = x.shape

    # Between-chain variance divided by n (B/n):
    # (mean per chain - global mean)^2 summed over chains, normalized by (m - 1)
    B_over_n = ((jnp.mean(x, axis=1) - jnp.mean(x))**2).sum() / (m_chains - 1)

    # Within-chain variance (W): average of chain variances
    # Compute per-chain deviations from their own means, square, sum, and normalize
    W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains*(n_iters - 1))

    # Over-dispersed variance estimator (s^2), a convex combination of W and B/n
    s2 = W * (n_iters - 1) / n_iters + B_over_n

    return s2

def compute_effective_sample_size_single_param(x):
    """
    Compute the effective sample size (ESS) for a single scalar parameter using multiple chains.

    Uses the initial monotone (Geyer) / initial positive sequence style truncation:
    we estimate lag-t autocorrelation via the variogram and stop when the sum of
    two consecutive autocorrelation estimates becomes negative.

    Parameters
    ----------
    x : array-like of shape (m_chains, n_iters)
        Samples for one scalar parameter across m_chains parallel chains.

    Returns
    -------
    ess : int
        Estimated effective sample size across all chains for this parameter.

    Notes
    -----
    - Autocorrelation at lag t is estimated via:
          rho[t] = 1 - variogram(t) / (2 * post_var)
      where `post_var` is the over-dispersed variance estimate from `gelman_rubin(x)`,
      and
          variogram(t) = mean over chains of mean over time of (x_{t:} - x_{:-t})^2.
    - Truncation rule: stop at the first even t for which rho[t-1] + rho[t] < 0.
    - Assumes equal-length chains, post-warmup samples, m_chains >= 2, n_iters >= 2.

    Examples
    --------
    >>> # x: (m_chains, n_iters)
    >>> ess = compute_effective_sample_size_single_param(x)
    >>> isinstance(ess, int)
    True
    """
    m_chains, n_iters = x.shape

    # Variogram at lag t: mean squared increment across chains/time at that lag
    variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    # Over-dispersed variance estimate (scalar)
    post_var = gelman_rubin(x)

    t = 1
    rho = np.ones(n_iters)  # rho[0] = 1 by definition; we fill rho[1:], then truncate
    negative_autocorr = False

    # Iterate over lags until the sum of consecutive autocorr estimates becomes negative
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        # Apply the "initial positive sequence" truncation on even t:
        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0

        t += 1

    # ESS across all chains: mn / (1 + 2 * sum_{t>=1} rho_t), truncated at the chosen t
    return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))

def compute_effective_sample_size(chains_):
    """
    Vectorized ESS for each parameter from multiple chains.

    Parameters
    ----------
    chains_ : array-like of shape (num_chains, num_samples, num_params)
        MCMC draws for `num_params` scalar parameters from `num_chains` parallel chains.

    Returns
    -------
    S_eff : ndarray of shape (num_params,)
        Effective sample size estimate for each parameter.

    Notes
    -----
    - Internally, each parameter slice `chains[:, :, idx_param]` (shape (num_chains, num_samples))
      is passed to `compute_effective_sample_size_single_param`.
    - Assumes equal-length chains and post-warmup samples.

    Examples
    --------
    >>> # 4 chains, 2000 samples, 3 parameters
    >>> # chains has shape (4, 2000, 3)
    >>> S_eff = compute_effective_sample_size(chains)
    >>> S_eff.shape
    (3,)
    """
    # force numpy array (function downstream uses NumPy ops)
    chains = np.array(chains_)

    # Unpack dimensions
    num_chains, num_samples, num_params = chains.shape

    # Compute ESS per parameter by slicing the last dimension
    S_eff = np.array([compute_effective_sample_size_single_param(chains[:, :, idx_param])
                      for idx_param in range(num_params)])

    return S_eff


def compute_Rhat(chains):
    """
    Compute the split-\u0052\u0068\u0061\u0074 (Gelman–Rubin) convergence diagnostic for each parameter.

    This variant splits each chain in half (to increase sensitivity to non-stationarity),
    computes within- and between-chain variances on the subchains, and returns:
        Rhat = sqrt(Var_plus / W),
    where
        Var_plus = ((n - 1)/n) * W + (1/n) * B,
    with n = original per-chain length (before splitting).

    Parameters
    ----------
    chains : array-like of shape (num_chains, num_samples, num_params)
        MCMC draws from multiple parallel chains for multiple parameters.

    Returns
    -------
    Rhat : ndarray of shape (num_params,)
        Split-\u0052\u0068\u0061\u0074 per parameter. Values close to 1.00 indicate good mixing.
        Common rules of thumb: < 1.1 (older), < 1.05 (stricter), or even < 1.01 in some settings.

    Notes
    -----
    - This is the classic split-\u0052\u0068\u0061\u0074. Modern recommendations (Vehtari et al., 2021)
      use rank-normalization and folding for more robustness; this function does not.
    - All chains must have the same length; warm-up should be removed prior to calling.
    - `num_samples` must be even to split cleanly in halves; if odd, the second half will be 1 sample longer.

    Examples
    --------
    >>> # 4 chains, 1000 samples, 2 parameters
    >>> # chains shape: (4, 1000, 2)
    >>> rhat = compute_Rhat(chains)
    >>> rhat.shape
    (2,)
    """
    # Unpack dimensions
    num_chains, num_samples, num_params = chains.shape

    # Split each chain into two subchains of approximately equal length
    sub_chains = []
    half_num_samples = int(0.5*num_samples)
    for idx_chain in range(num_chains):
        sub_chains.append(chains[idx_chain, :half_num_samples, :])   # first half
        sub_chains.append(chains[idx_chain, half_num_samples:, :])   # second half

    # Count subchains (should be 2 * num_chains)
    num_sub_chains = len(sub_chains)
        
    # Compute mean and variance per subchain (vectorized over parameters)
    # chain_means: (num_sub_chains, num_params)
    chain_means = np.array([np.mean(s, axis=0) for s in sub_chains])
    # chain_vars: (num_sub_chains, num_params), unbiased within-subchain variance
    chain_vars = np.array([1/(num_samples-1)*np.sum((s-m)**2, 0) for (s, m) in zip(sub_chains, chain_means)])

    # Between-subchain variance B: measures dispersion of subchain means
    # global_mean: (num_params,)
    global_mean = np.mean(chain_means, axis=0)
    B = num_samples/(num_sub_chains-1)*np.sum((chain_means - global_mean)**2, axis=0)  # (num_params,)

    # Within-subchain variance W: average of subchain variances
    W = np.mean(chain_vars, 0)  # (num_params,)

    # Pooled variance estimator Var_plus and split-Rhat
    var_estimator = (num_samples-1)/num_samples*W + (1/num_samples)*B  # (num_params,)
    Rhat = np.sqrt(var_estimator/W)
    return Rhat


def combine_chains(chains):
    """
    Flatten an array of MCMC draws into a 1D vector.

    Parameters
    ----------
    chains : array-like
        Any shape. Commonly, MCMC outputs are shaped as:
        - (num_chains, num_samples) for a single parameter, or
        - (num_chains, num_samples, num_params) for multiple parameters.

    Returns
    -------
    flattened : ndarray of shape (np.prod(chains.shape),)
        A 1D view/copy (implementation dependent) of the input samples.

    Notes
    -----
    - This simply calls `.flatten()` on the input. It does not concatenate chains along
      a particular axis while preserving parameter boundaries; it discards structure.
    - If you intend to *stack* chains along the sample axis while preserving the parameter
      dimension (e.g., to form (num_chains*num_samples, num_params)), do that explicitly
      with `reshape`/`transpose` instead of using this helper.

    Examples
    --------
    >>> # For a single parameter with shape (4, 1000)
    >>> vec = combine_chains(x)           # shape (4000,)
    >>> # For multiple parameters (4, 1000, 3)
    >>> vec = combine_chains(chains)      # shape (12000,)
    """
    return chains.flatten()