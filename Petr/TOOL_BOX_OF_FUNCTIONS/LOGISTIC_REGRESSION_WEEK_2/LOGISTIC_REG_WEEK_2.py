import jax.numpy as jnp
from jax.scipy.stats import binom  # Note: you'll need this import for binom_dist.logpmf

import jax.numpy as jnp
from jax import random

#! Sigmoid activation function: σ(x) = 1 / (1 + exp(-x))
#! Maps any real number to (0,1), turning raw linear predictions into probabilities
#! Used for binary classification in logistic regression
#! Input shape: (...) -> Output shape: (...)
sigmoid = lambda x: 1./(1 + jnp.exp(-x))

#! Log probability density function for normal distribution
#! log N(x; μ, σ²) = -(x-μ)² / (2σ²) - 0.5 * log(2πσ²)
#! Used for evaluating the log prior for Gaussian priors
#! Input shapes: x(...), m(...), v(...) -> Output shape: (...)
log_npdf = lambda x, m, v: -(x-m)**2/(2*v) - 0.5*jnp.log(2*jnp.pi*v)


class LogisticRegression(object):
    """
    Bayesian Logistic Regression for binomial data.
    
    Models the probability of success as:
        f(x)      = α + βx                            (linear predictor)    (eq. 3)
        θ(x)      = σ(f(x))                           (sigmoid probability) (eq. 2)
        y_i       ~ Binomial(N_i, θ(x_i))             (likelihood)          (eq. 5)
        α, β      ~ N(0, σ²_α), N(0, σ²_β)             (priors)             (eq. 8)


    Use case: Binary classification or proportion modeling where you want
    uncertainty quantification over parameters α and β.
    Example use:
        >>> model = LogisticRegression(x, y, N)
        >>> model.log_joint(alpha, beta)  # For posterior computation
    """
    
    def __init__(self, x, y, N, sigma2_alpha=1., sigma2_beta=1.):
        """
        Initialize the logistic regression model.
        
        Args:
            x: Input features, shape (n_samples,)
            y: Observed successes (counts), shape (n_samples,) 
            N: Number of trials for each observation, shape (n_samples,) or scalar
            sigma2_alpha: Prior variance for intercept α, scalar
            sigma2_beta: Prior variance for slope β, scalar
            
        Example:
            x = jnp.array([1.0, 2.0, 3.0])     # 3 data points
            y = jnp.array([2, 5, 8])           # successes observed
            N = jnp.array([10, 10, 10])        # trials per data point
        """
        # Data storage
        self.x = x                    # Features: shape (n_samples,)
        self.y = y                    # Success counts: shape (n_samples,) 
        self.N = N                    # Trial counts: shape (n_samples,) or scalar
        
        # Hyperparameters for Bayesian priors
        self.sigma2_alpha = sigma2_alpha    # Prior variance for intercept α
        self.sigma2_beta = sigma2_beta      # Prior variance for slope β
    
    def f(self, x, alpha, beta):
        """
        Linear function: f(x) = α + βx                                    (eq. 3)
        
        This is the linear predictor before sigmoid transformation.
        
        Args:
            x: Input features, shape (..., n_features)
            alpha: Intercept parameter, shape (..., 1) or broadcastable
            beta: Slope parameter, shape (..., 1) or broadcastable
            
        Returns:
            Linear predictions, same shape as x
            
        Note: Output must have the same shape as x for vectorized operations
        """
        return alpha + beta * x
    
    def theta(self, x, alpha, beta):
        """
        Success probability: θ(x) = σ(f(x)) = σ(α + βx)                  (eq. 2)
        
        Applies sigmoid to convert linear predictions to probabilities ∈ (0,1).
        
        Args:
            x: Input features, shape (..., n_features)  
            alpha: Intercept parameter, shape (..., 1) or broadcastable
            beta: Slope parameter, shape (..., 1) or broadcastable
            
        Returns:
            Success probabilities, same shape as x, values ∈ (0,1)
        """
        return sigmoid(self.f(x, alpha, beta))
    
    def log_prior(self, alpha, beta):
        """
        Log prior probability: log p(α, β)                               (eq. 8)
        
        Assumes independent normal priors:
        - α ~ N(0, σ²_α)  
        - β ~ N(0, σ²_β)
        - log p(α, β) = log N(α; 0, σ²_α) + log N(β; 0, σ²_β)
        
        Args:
            alpha: Intercept parameter, shape (..., 1)
            beta: Slope parameter, shape (..., 1)
            
        Returns:
            Log prior probability, shape (..., 1)
            
        Use case: Regularization - penalizes large parameter values
        """
        log_p_alpha = log_npdf(alpha, 0, self.sigma2_alpha)
        log_p_beta = log_npdf(beta, 0, self.sigma2_beta) 
        return log_p_alpha + log_p_beta
    
    def log_likelihood(self, alpha, beta):
        """
        Log likelihood: log p(y | x, α, β, N)                           (eq. 5)
        
        Assumes binomial likelihood for each data point:
        y_i ~ Binomial(N_i, θ(x_i)) where θ(x_i) = σ(α + β*x_i)
        
        Total log likelihood = Σᵢ log Binomial(y_i; N_i, θ(x_i))
        
        Args:
            alpha: Intercept parameter, shape (..., 1) 
            beta: Slope parameter, shape (..., 1)
            
        Returns:
            Log likelihood, shape (..., 1)
            
        Note: Uses vectorized operations across all data points
        """
        # Compute success probabilities for all data points
        theta = self.theta(self.x, alpha, beta)  # shape: (..., n_samples)
        
        # Compute log probability for each binomial observation
        # binom.logpmf(k, n, p) = log P(X = k | X ~ Binomial(n, p))
        log_lik = jnp.sum(
            binom.logpmf(self.y, n=self.N, p=theta), 
            axis=-1, 
            keepdims=True
        )
        return log_lik
    
    def log_joint(self, alpha, beta):
        """
        Log joint probability: log p(α, β, y | x, N)
        
        By Bayes' theorem:
        log p(α, β | y, x, N) ∝ log p(y | x, α, β, N) + log p(α, β)
        
        Args:
            alpha: Intercept parameter, shape (..., 1)
            beta: Slope parameter, shape (..., 1)
            
        Returns:
            Log joint probability, shape (...,)
            
        Use case: 
        - Optimization target for MAP estimation
        - Unnormalized log posterior for MCMC sampling
        - Model comparison via marginal likelihood
        """
        log_prior = self.log_prior(alpha, beta).squeeze()
        log_likelihood = self.log_likelihood(alpha, beta).squeeze() 
        return log_prior + log_likelihood




class Grid2D(object):
    """
    Helper class for evaluating a function func on a 2D grid defined by (alpha, beta) parameters.
    
    Purpose: Enables visualization and analysis of probability distributions over parameter space
    for Bayesian inference in logistic regression.
    
    Mathematical Context:
    - For logistic regression: P(y|x) = sigmoid(alpha + beta*x)
    - We want to analyze: P(alpha, beta | data) ∝ P(data | alpha, beta) * P(alpha, beta)
    """

    def __init__(self, alphas, betas, func, name="Grid2D"):
        """
        Initialize the 2D grid and evaluate the function at all grid points.
        
        Args:
            alphas: 1D array of alpha values, shape (n_alpha,)
            betas: 1D array of beta values, shape (n_beta,) 
            func: Function to evaluate, takes (alpha, beta) and returns scalar or array
            name: String identifier for plotting
        """
        self.alphas = alphas                    # Shape: (n_alpha,)
        self.betas = betas                      # Shape: (n_beta,)
        self.grid_size = (len(self.alphas), len(self.betas))  # (n_alpha, n_beta) 
        # self.grid_size stores (J, K) so later we can map a 1-D index back to 2-D coordinates (needed for argmax)
        
        # Create 2D coordinate matrices for vectorized function evaluation
        # indexing='ij' means alpha varies along rows, beta along columns
        self.alpha_grid, self.beta_grid = jnp.meshgrid(alphas, betas, indexing='ij')
        # Shapes: alpha_grid (n_alpha, n_beta), beta_grid (n_alpha, n_beta)
        # Note: This produces two (J, K) arrays, A_jk = alphas[j] and B_jk = betas[k]
        
        self.func = func # Function to evaluate on the grid e.g. log_prior, log_likelihood, log_joint
        self.name = name # Identifier for the grid, used in plots, e.g. 'Prior', 'Likelihood', 'Posterior'
        
        # Evaluate function on each grid point
        # Add None dimension for broadcasting, then squeeze to remove singleton dims
        # Input shapes: (n_alpha, n_beta, 1) each
        # Output shape after squeeze: (n_alpha, n_beta)
        self.values = self.func(self.alpha_grid[:, :, None], self.beta_grid[:, :, None]).squeeze()
        # Note: Broadcasting trick: [:, :, None] adds a singleton 3ʳᵈ dimension so the model 
        # functions—written to accept shapes (batch, 1) or (1, batch)—work unchanged.
        # .squeeze() removes that dummy dimension → a plain (J,K)(J,K) table.

    def plot_contours(self, ax, color='b', num_contours=10, f=lambda x: x, alpha=1.0, title=None):
        """
        Plot contour lines of the evaluated function.
        
        Args:
            ax: Matplotlib axes object
            color: Color for contour lines
            num_contours: Number of contour levels
            f: Transform function (e.g., exp to convert log-space to probability space)
            alpha: Transparency level
            title: Optional plot title override
            
        Note: .T transpose is needed because matplotlib expects (y, x) indexing
        while our grid uses (alpha, beta) = (x, y) indexing
        """
        ax.contour(self.alphas, self.betas, f(self.values).T, num_contours, colors=color, alpha=alpha)
        ax.set(xlabel='$\\alpha$', ylabel='$\\beta$')
        ax.set_title(self.name, fontweight='bold')

    @property
    def argmax(self):
        """
        Find the (alpha, beta) coordinates of the maximum value.
        
        Returns:
            tuple: (alpha_max, beta_max) coordinates
            
        Mathematical meaning:
        - For log-likelihood: gives MLE (Maximum Likelihood Estimate)
        - For log-posterior: gives MAP (Maximum A Posteriori) estimate
        """
        idx = jnp.argmax(self.values)  # Flattened index of maximum
        # Convert flat index back to 2D coordinates
        alpha_idx, beta_idx = jnp.unravel_index(idx, self.grid_size)
        return self.alphas[alpha_idx], self.betas[beta_idx]


class GridApproximation2D(Grid2D):

    def __init__(self, alphas, betas, log_joint, threshold=1e-8, name="GridApproximation2D"):
        Grid2D.__init__(self, alphas, betas, log_joint, name)
        self.threshold = threshold
        self.prep_approximation()
        self.compute_marginals()
        self.sanity_check()
        
    def prep_approximation(self):
        
        # [num_alpha, num_beta]-sized matrix of the log joint evaluated on the grid 
        self.log_joint_grid = self.values
        self.log_joint_grid = self.log_joint_grid - jnp.max(self.log_joint_grid)

        # evaluate joint for each point on the grid
        self.tilde_probabilities_grid = jnp.exp(self.log_joint_grid) 

        # compute normalization constant
        self.Z = jnp.sum(self.tilde_probabilities_grid)      

        # [num_alpha, num_beta]-matrix of \pi_{ij}-values summing to 1.
        self.probabilities_grid = self.tilde_probabilities_grid/self.Z    

        # flatten for later convinience
        self.alphas_flat = self.alpha_grid.flatten()                                             # shape: [num_alpha*num_beta] = [num_outcomes]
        self.betas_flat = self.beta_grid.flatten()                                               # shape: [num_alpha*num_beta] = [num_outcomes]
        self.num_outcomes = len(self.alphas_flat)                                                # shape: scalar 
        self.probabilities_flat = self.probabilities_grid.flatten()                              # [num_outcomes]

    def compute_marginals(self):
        self.pi_alpha = self.probabilities_grid.sum(1)  
        self.pi_beta = self.probabilities_grid.sum(0)  

        # compute marginal distribution using sum rule
    def compute_expectation(self, f):
        """ computes expectation of f(alpha, beta) wrt. the grid approximation """
        return jnp.sum(f(self.alphas_flat, self.betas_flat)*self.probabilities_flat, axis=0)
    
    def sample(self, key, num_samples=1):
        """ generate num_samples from the grid approximation distribution """
        idx = random.choice(key, jnp.arange(self.num_outcomes), p=self.probabilities_flat, shape=(num_samples, 1))
        return self.alphas_flat[idx], self.betas_flat[idx]

    def visualize(self, ax, scaling=8000, title='Grid approximation'):
        idx = self.probabilities_flat > self.threshold
        ax.scatter(self.alphas_flat[idx], self.betas_flat[idx], scaling*self.probabilities_flat[idx],label='$\\pi_{ij}$')        
        ax.set(xlabel='$\\alpha$', ylabel='$\\beta$')
        ax.set_title(title, fontweight='bold')

    def sanity_check(self):
        assert self.probabilities_grid.shape == self.grid_size, "Probability grid does not have shape [num_alphas, num_betas] (self.grid_size). Check your implementation."
        assert jnp.all(self.probabilities_grid >= 0), "Not all values in probability grid are non-negative. Check your implementation."
        assert jnp.allclose(self.probabilities_grid.sum(), 1), "Values in probability grid do not sum to one. Check your implementation."


class DiscreteDistribution1D(object):

    def __init__(self, outcomes, probabilities, name='DiscreteDistribution'):
        """ represents discrete random variable X in terms of outcomes and probabilities """
        self.outcomes = outcomes
        self.probabilities = probabilities
        assert self.outcomes.shape == self.probabilities.shape
        self.name = name

    def CDF(self, x):
        """ P[X <= x] """
        idx = self.outcomes <= x
        return jnp.sum(self.probabilities[idx]) 
    
    def quantile(self, p):
        """ Q(p) = inf {x | p < CDF(x)} """
        cdf_values = jnp.cumsum(self.probabilities) 
        idx = jnp.where(jnp.logical_or(p < cdf_values, jnp.isclose(p, cdf_values)))[0]
        return jnp.min(self.outcomes[idx])
    
    @property
    def mean(self):
        """ return scalar corresponding to the mean of the discrete distribution """
        return jnp.mean(self.outcomes)
    
    @property
    def variance(self):
        """ return scalar corresponding to the variance of the discrete distribution """
        return jnp.sum((self.outcomes - self.mean)**2 * self.probabilities) 
    
    def central_interval(self, interval_size=95):
        """ return tuple (lower, upper) corresponding to the central interval of the discrete distribution """
        c = 1.-interval_size/100.
        lower = self.quantile(c/2)
        upper = self.quantile(1-c/2)
        return jnp.array([lower, upper])  
    
    def print_summary(self):
        print(f'Summary for {self.name}')
        print(f'\tMean:\t\t\t\t{self.mean:3.2f}')
        print(f'\tStd. dev.:\t\t\t{jnp.sqrt(self.variance):3.2f}')
        print(f'\t95%-credibility interval:\t[{self.central_interval()[0]:3.2f}, {self.central_interval()[1]:3.2f}]\n')


    