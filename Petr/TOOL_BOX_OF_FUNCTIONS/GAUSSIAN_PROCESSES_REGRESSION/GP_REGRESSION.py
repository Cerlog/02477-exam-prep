import jax.numpy as jnp
from jax import value_and_grad
from jax import random
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import seaborn as snb


from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


# we want to use 64 bit floating precision
import jax
jax.config.update("jax_enable_x64", True)

snb.set_style('darkgrid')
snb.set_theme(font_scale=1.25)


import jax.numpy as jnp
from jax import random

def generate_samples(key, m, K, num_samples, jitter=1e-8):
    r"""
    Draw samples from a multivariate Gaussian :math:`\mathcal{N}(m, K)` using the
    Cholesky factorization with optional diagonal jitter for numerical stability.

    **Mathematics**

    Let :math:`L` be the lower-triangular Cholesky factor of
    :math:`K + \epsilon I`, i.e.
    :math:`L L^\top = K + \epsilon I`.

    If :math:`z \sim \mathcal{N}(0, I)`, then

    .. math::

        f = m + L z \sim \mathcal{N}(m, K + \epsilon I).

    When `num_samples = M`, we sample a matrix
    :math:`Z \in \mathbb{R}^{N \times M}` whose columns are i.i.d.
    :math:`\mathcal{N}(0, I)`, and return

    .. math::

        F = m \mathbf{1}^\top + L Z \in \mathbb{R}^{N \times M}.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key controlling the RNG (reproducibility).
    m : array_like, shape (N,)
        Mean vector of the Gaussian.
    K : array_like, shape (N, N)
        Covariance (kernel) matrix.
    num_samples : int
        Number of samples :math:`M` to generate.
    jitter : float, optional (default=1e-8)
        Non-negative scalar added to the diagonal of `K` before the Cholesky
        factorization to ensure positive definiteness numerically.

    Returns
    -------
    f_samples : jnp.ndarray, shape (N, M)
        Matrix whose columns are i.i.d. samples from
        :math:`\mathcal{N}(m, K + \epsilon I)`.

    Notes
    -----
    - If you need a significantly large `jitter` (e.g. > 1e-3 relative to the
      scale of `K`), that often indicates numerical pathologies (duplicate inputs,
      extreme hyperparameters, etc.).
    - Shapes:
        * `m`: (N,)
        * `K`: (N, N)
        * `Z`: (N, M)
        * `L`: (N, N)
        * `f_samples`: (N, M)

    Examples
    --------
    >>> key = random.PRNGKey(1)
    >>> num_samples = 100_000
    >>> m = jnp.array([jnp.pi, jnp.sqrt(2.0)])
    >>> V = jnp.array([[0.123, -0.05], [-0.05, 0.123]])
    >>> f_samples = generate_samples(key, m, V, num_samples)
    >>> # Check sample mean and covariance are close to the target:
    >>> assert jnp.linalg.norm(jnp.mean(f_samples, axis=1) - m) < 1e-2
    >>> assert jnp.linalg.norm(jnp.cov(f_samples) - V) < 1e-2
    """
    # Convert to JAX arrays (no-ops if already so)
    m = jnp.asarray(m)
    K = jnp.asarray(K)

    # -------------------------
    # Sanity checks on shapes
    # -------------------------
    assert m.ndim == 1, f"`m` must be a 1-D vector, got shape {m.shape}"
    assert K.ndim == 2 and K.shape[0] == K.shape[1], \
        f"`K` must be a square matrix, got shape {K.shape}"
    N = K.shape[0]
    assert m.shape[0] == N, \
        f"Length of `m` ({m.shape[0]}) must match K.shape[0] ({N})"
    assert num_samples > 0, "`num_samples` must be a positive integer"
    assert jitter >= 0, "`jitter` must be non-negative"

    # ---------------------------------------
    # 1) Sample Z ~ N(0, I), shape (N, M)
    # ---------------------------------------
    Z = random.normal(key, shape=(N, num_samples))

    # ---------------------------------------
    # 2) Cholesky of K + epsilon I
    #    L has shape (N, N)
    # ---------------------------------------
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(N))

    # ---------------------------------------
    # 3) f = m[:, None] + L @ Z
    #    m[:, None] broadcasts to (N, M)
    # ---------------------------------------
    f_samples = m[:, None] + L @ Z

    # ---------------------------------------
    # 4) Final shape check
    # ---------------------------------------
    assert f_samples.shape == (N, num_samples), \
        (f"The shape of f_samples appears wrong. Expected shape ({N}, {num_samples}), "
         f"but got {f_samples.shape}. Please check your code.")

    return f_samples


# in the code below tau represents the distance between to input points, i.e. tau = ||x_n - x_m||.
def squared_exponential(tau, kappa, lengthscale):
    return kappa**2*jnp.exp(-0.5*tau**2/lengthscale**2) 

def matern12(tau, kappa, lengthscale):
    return kappa**2*jnp.exp(-tau/lengthscale)

def matern32(tau, kappa, lengthscale):
    return kappa**2*(1 + jnp.sqrt(3)*tau/lengthscale)*jnp.exp(-jnp.sqrt(3)*tau/lengthscale)


class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun, kappa=1., lengthscale=1.0):
        """
            the argument kernel_fun must be a function of three arguments kernel_fun(||tau||, kappa, lengthscale), e.g. 
            squared_exponential = lambda tau, kappa, lengthscale: kappa**2*np.exp(-0.5*tau**2/lengthscale**2)
        """
        self.kernel_fun = kernel_fun
        self.kappa = kappa
        self.lengthscale = lengthscale

    def contruct_kernel(self, X1, X2, kappa=None, lengthscale=None, jitter=1e-8):
        """ compute and returns the NxM kernel matrix between the two sets of input X1 (shape NxD) and X2 (MxD) using the stationary and isotropic covariance function specified by self.kernel_fun
    
        arguments:
            X1              -- NxD matrix
            X2              -- MxD matrix
            kappa           -- magnitude (positive scalar)
            lengthscale     -- characteristic lengthscale (positive scalar)
            jitter          -- non-negative scalar
        
        returns
            K               -- NxM matrix    
        """

        # extract dimensions 
        N, M = X1.shape[0], X2.shape[0]

        # prep hyperparameters
        kappa = self.kappa if kappa is None else kappa
        lengthscale = self.lengthscale if lengthscale is None else lengthscale

        ##############################################
        # Your solution goes here
        ##############################################
        # compute all the pairwise distances efficiently
        dist = jnp.sqrt( jnp.sum( (jnp.expand_dims(X1, axis=1) - jnp.expand_dims(X2, axis=0))**2, axis=-1 ) )
        
        # squared exponential covariance function 
        K = self.kernel_fun(dist, kappa, lengthscale)
        
        # add jitter to diagonal for numerical stability 
        if len(X1) == len(X2) and jnp.allclose(X1, X2):
            K = K + jitter * jnp.identity( len(X1) )
        ##############################################
        # End of solution
        ##############################################
        
        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but the actual shape was {K.shape}. Please check your code. "
        return K


def plot_with_uncertainty(ax, Xp, gp, color='r', color_samples='b', title="", num_samples=0, seed=0):
    
    mu, Sigma = gp.predict_y(Xp)
    mean = mu.ravel()
    std = jnp.sqrt(jnp.diag(Sigma))

    # random seed
    key = random.PRNGKey(seed)

    # plot distribution
    ax.plot(Xp, mean, color=color, label='Mean')
    ax.plot(Xp, mean + 2*std, color=color, linestyle='--')
    ax.plot(Xp, mean - 2*std, color=color, linestyle='--')
    ax.fill_between(Xp.ravel(), mean - 2*std, mean + 2*std, color=color, alpha=0.25, label='95% interval')
    
    # generate samples
    if num_samples > 0:
        fs = gp.posterior_samples(key, Xstar, num_samples)
        ax.plot(Xp, fs[:,0], color=color_samples, alpha=.25, label="$f(x)$ samples")
        ax.plot(Xp, fs[:, 1:], color=color_samples, alpha=.25)
    
    ax.set_title(title)
    



class GaussianProcessRegression(object):

    def __init__(self, X, y, kernel, kappa=1., lengthscale=1., sigma=1/2, jitter=1e-8):
        """  
        Arguments:
            X                -- NxD input points
            y                -- Nx1 observed values 
            kernel           -- must be instance of the StationaryIsotropicKernel class
            jitter           -- non-negative scaler
            kappa            -- magnitude (positive scalar)
            lengthscale      -- characteristic lengthscale (positive scalar)
            sigma            -- noise std. dev. (positive scalar)
        """
        self.X = X
        self.y = y
        self.N = len(X)
        self.kernel = kernel
        self.jitter = jitter
        self.set_hyperparameters(kappa, lengthscale, sigma)
        self.check_dimensions()

    def check_dimensions(self):
        assert self.X.ndim == 2, f"The variable X must be of shape (N, D), however, the current shape is: {self.X.shape}"
        N, D = self.X.shape

        assert self.y.ndim == 2, f"The varabiel y must be of shape (N, 1), however. the current shape is: {self.y.shape}"
        assert self.y.shape == (N, 1), f"The varabiel y must be of shape (N, 1), however. the current shape is: {self.y.shape}"
        

    def set_hyperparameters(self, kappa, lengthscale, sigma):
        self.kappa = kappa
        self.lengthscale = lengthscale
        self.sigma = sigma

    def posterior_samples(self, key, Xstar, num_samples):
        """
            generate samples from the posterior p(f^*|y, x^*) for each of the inputs in Xstar

            Arguments:
                key              -- jax random key for controlling the random number generator
                Xstar            -- PxD prediction points
        
            returns:
                f_samples        -- numpy array of (P, num_samples) containing num_samples for each of the P inputs in Xstar
        """
        ##############################################
        # Your solution goes here
        ##############################################
        
        mu, Sigma = self.predict_f(Xstar)
        f_samples = generate_samples(key, mu.ravel(), Sigma, num_samples)
        
        ##############################################
        # End of solution
        ##############################################

        assert (f_samples.shape == (len(Xstar), num_samples)), f"The shape of the posterior mu seems wrong. Expected ({len(Xstar)}, {num_samples}), but actual shape was {f_samples.shape}. Please check implementation"
        return f_samples
        
    def predict_y(self, Xstar):
        """ returns the posterior distribution of y^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Arguments:
        Xstar            -- PxD prediction points
        
        returns:
        mu               -- Px1 mean vector
        Sigma            -- PxP covariance matrix
        """

        ##############################################
        # Your solution goes here
        ##############################################
        
        # prepare relevant matrices
        mu, Sigma = self.predict_f(Xstar)
        Sigma = Sigma + self.sigma**2 * jnp.identity(len(mu))
        
        ##############################################
        # End of solution
        ##############################################

        return mu, Sigma

    def predict_f(self, Xstar):
        """ returns the posterior distribution of f^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Arguments:
        Xstar            -- PxD prediction points
        
        returns:
        mu               -- Px1 mean vector
        Sigma            -- PxP covariance matrix
        """

        ##############################################
        # Your solution goes here
        ##############################################
        
        # prepare relevant matrices
        k = self.kernel.contruct_kernel(Xstar, self.X, self.kappa, self.lengthscale, jitter=self.jitter)
        K = self.kernel.contruct_kernel(self.X, self.X, self.kappa, self.lengthscale, jitter=self.jitter)
        Kstar = self.kernel.contruct_kernel(Xstar, Xstar, self.kappa, self.lengthscale, jitter=self.jitter)
        
        # Compute C matrix
        C = K + self.sigma**2*jnp.identity(len(self.X)) 

        # computer mean and Sigma
        mu = jnp.dot(k, jnp.linalg.solve(C, self.y))
        Sigma = Kstar - jnp.dot(k, jnp.linalg.solve(C, k.T))
        
        ##############################################
        # End of solution
        ##############################################

        # sanity check for dimensions
        assert (mu.shape == (len(Xstar), 1)), f"The shape of the posterior mu seems wrong. Expected ({len(Xstar)}, 1), but actual shape was {mu.shape}. Please check implementation"
        assert (Sigma.shape == (len(Xstar), len(Xstar))), f"The shape of the posterior Sigma seems wrong. Expected ({len(Xstar)}, {len(Xstar)}), but actual shape was {Sigma.shape}. Please check implementation"

        return mu, Sigma
    
    def log_marginal_likelihood(self, kappa, lengthscale, sigma):
        """ 
            evaluate the log marginal likelihood p(y) given the hyperparaemters 

            Arguments:
            kappa       -- positive scalar 
            lengthscale -- positive scalar
            sigma       -- positive scalar
            """

        ##############################################
        # Your solution goes here
        ##############################################
        
        # prepare kernels
        K = self.kernel.contruct_kernel(self.X, self.X, kappa, lengthscale)
        C = K + sigma**2*jnp.identity(self.N)

        # compute Cholesky decomposition
        L = jnp.linalg.cholesky(C)
        v = jnp.linalg.solve(L, self.y)

        # compute log marginal likelihood
        logdet_term = jnp.sum(jnp.log(jnp.diag(L)))
        quad_term =  0.5*jnp.sum(v**2)
        const_term = -0.5*self.N*jnp.log(2*jnp.pi)

        return const_term - logdet_term - quad_term
        
        ##############################################
        # End of solution
        ##############################################
        
            
    def prior_predictive_f_star(self, Xstar):
            """
            Computes the prior predictive distribution for f^* at the test points Xstar.

            Arguments:
                Xstar -- PxD array of prediction points

            Returns:
                mu0   -- Px1 mean vector (all zeros, since prior mean is zero)
                Sigma0-- PxP covariance matrix (prior covariance for f^*)

            Mathematical details:
                - Prior mean:   mu0 = 0
                - Prior covariance: Sigma0 = K(Xstar, Xstar)
                - Shapes: mu0.shape = (P, 1), Sigma0.shape = (P, P)
            """
            K_star = self.kernel.contruct_kernel(Xstar, Xstar, self.kappa, self.lengthscale)
            mu0 = jnp.zeros((len(Xstar), 1))
            Sigma0 = K_star 
            return mu0, Sigma0

    def prior_predictive_y_star(self, Xstar):
            """
            Computes the prior predictive distribution for y^* at the test points Xstar,
            including observation noise.

            Arguments:
                Xstar -- PxD array of prediction points

            Returns:
                mu0   -- Px1 mean vector (all zeros, since prior mean is zero)
                Sigma0-- PxP covariance matrix (prior covariance for y^*)

            Mathematical details:
                - Prior mean:   mu0 = 0
                - Prior covariance: Sigma0 = K(Xstar, Xstar) + sigma^2 * I
                - Shapes: mu0.shape = (P, 1), Sigma0.shape = (P, P)
                - Equation: y^* ~ N(0, K(Xstar, Xstar) + sigma^2 * I)
            """
            K_star = self.kernel.contruct_kernel(Xstar, Xstar, self.kappa, self.lengthscale)
            mu0 = jnp.zeros((len(Xstar), 1))
            Sigma0 = K_star + self.sigma**2 * jnp.identity(len(Xstar))
            return mu0, Sigma0

def optimize_hyperparameters(gp, theta_init):
    """
    Optimize GP hyperparameters by (approximately) maximizing the log marginal likelihood (LML).

    The parameter vector is θ = (κ, ℓ, σ) = (kappa, scale, sigma). To enforce positivity,
    this routine optimizes in log‑space: φ = log θ, and maps back via θ = exp(φ).

    Parameters
    ----------
    gp : GaussianProcessRegression
        A GP model that exposes:
          - gp.log_marginal_likelihood(kappa, scale, sigma) -> float
            (returns the *log* marginal likelihood, not its negative).
        The method is evaluated on the model's *current training data*.
        Make sure `gp` was constructed with your (Xtrain, ytrain).

    theta_init : jnp.ndarray, shape (3,)
        Positive initial guess for (κ, ℓ, σ). Example: jnp.array([1.0, 1.0, 1.0]).
        Ordering is strictly (kappa, scale, sigma).

    Returns
    -------
    theta : jnp.ndarray, shape (3,)
        The optimized hyperparameters (κ̂, ℓ̂, σ̂) in *original* space (positive values).

    Optimization target
    -------------------
    We minimize the negative LML:
        objective(φ) = - log p(y | X, θ=exp(φ))
    where φ := log θ. Gradients (value_and_grad) are taken w.r.t. φ.

    Notes
    -----
    - This function assumes availability of `value_and_grad` (from JAX) and
      `minimize` (e.g., SciPy). Any gradient‑based optimizer that accepts a
      value‑and‑gradient callable is suitable.
    - If your y has shape (n,1), the LML implementation should internally treat it as
      a vector of length n (e.g., squeeze) or handle the shape consistently.
    - Check `res.success`; if False, consider re‑trying with different `theta_init`,
      bounds, or a different optimizer.

    Example
    -------
    >>> kernel = StationaryIsotropicKernel(kernel_fun=squared_exponential)
    >>> gp_post = GaussianProcessRegression(Xtrain, ytrain, kernel)
    >>> theta0 = jnp.array([1.0, 1.0, 1.0])  # (kappa, lengthscale, noise)
    >>> theta_hat = optimize_hyperparameters(gp_post, theta0)
    >>> gp_post.set_hyperparameters(*theta_hat)  # use (κ̂, ℓ̂, σ̂)
    """
    # define optimization objective as the negative log marginal likelihood (minimize)
    objective = lambda params: -gp.log_marginal_likelihood(
        jnp.exp(params[0]),  # kappa = exp(φ_0) > 0
        jnp.exp(params[1]),  # scale (lengthscale) = exp(φ_1) > 0
        jnp.exp(params[2])   # sigma (noise std)  = exp(φ_2) > 0
    )

    # optimize using gradients wrt the *log* parameters
    res = minimize(value_and_grad(objective), jnp.log(theta_init), jac=True)

    # check for success
    if not res.success:
        print('Warning: optimization failed!')

    # return results in original space θ = exp(φ)
    theta = jnp.exp(res.x)
    return theta
