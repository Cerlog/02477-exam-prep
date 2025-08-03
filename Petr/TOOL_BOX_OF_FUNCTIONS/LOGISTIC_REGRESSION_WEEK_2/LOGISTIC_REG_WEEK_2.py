import jax.numpy as jnp
from jax.scipy.stats import binom  # Note: you'll need this import for binom_dist.logpmf

# Sigmoid activation function: σ(x) = 1/(1 + exp(-x))
# Maps any real number to (0,1), used to convert linear outputs to probabilities
# Input shape: (...) -> Output shape: (...)
sigmoid = lambda x: 1./(1 + jnp.exp(-x))

# Log probability density function for normal distribution
# log N(x; μ, σ²) = -(x-μ)²/(2σ²) - 0.5*log(2πσ²)
# Input shapes: x(...), m(...), v(...) -> Output shape: (...)
log_npdf = lambda x, m, v: -(x-m)**2/(2*v) - 0.5*jnp.log(2*jnp.pi*v)


class LogisticRegression(object):
    """
    Bayesian Logistic Regression for binomial data.
    
    Model assumptions:
    - Linear relationship: f(x) = α + βx                    (eq. 3)
    - Sigmoid transformation: θ(x) = σ(f(x))               (eq. 2) 
    - Binomial likelihood: y ~ Binomial(N, θ(x))           (eq. 5)
    - Normal priors: α ~ N(0, σ²_α), β ~ N(0, σ²_β)       (eq. 8)
    
    Use case: Binary classification or proportion modeling where you want
    uncertainty quantification over parameters α and β.
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
