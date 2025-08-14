import jax.numpy as jnp
from jax import value_and_grad
from jax import random
from scipy.optimize import minimize

def design_matrix(x):
    """
    Construct a design matrix Φ with a bias (intercept) term.

    Parameters:
    - x: (n,) input array

    Returns:
    - Φ: (n, 2) design matrix with ones in first column and x values in second

    Example:
    >>> x = jnp.array([1., 2., 3.])
    >>> design_matrix(x)
    DeviceArray([[1., 1.],
                 [1., 2.],
                 [1., 3.]], dtype=float32)
    """
    return jnp.column_stack((jnp.ones(len(x)), x))

class BayesianLinearRegression(object):
    """
    Bayesian Linear Regression Implementation
    
    This class implements Bayesian linear regression with Gaussian priors and likelihood.
    The model assumes:
    - Prior: w ~ N(0, α^(-1)I) where α is the precision (inverse variance) of weights
    - Likelihood: y = Φw + ε, where ε ~ N(0, β^(-1)) and β is noise precision
    - Posterior: w|y ~ N(m, S) where m and S are computed analytically
    
    Mathematical Framework:
    =====================
    Given training data (Φ, y) where Φ is [N×D] design matrix and y is [N×1] targets:
    
    Prior Distribution:
    p(w) = N(w | 0, α^(-1)I)
    
    Likelihood:
    p(y | w, Φ, β) = N(y | Φw, β^(-1)I)
    
    Posterior Distribution:
    p(w | y, Φ, α, β) = N(w | m, S)
    
    Where:
    S^(-1) = αI + βΦ^T Φ                    (posterior precision matrix)
    S = (αI + βΦ^T Φ)^(-1)                  (posterior covariance matrix)
    m = βS Φ^T y                            (posterior mean)
    
    Predictive Distribution:
    For new input Φ*, the predictive distribution of function values f* = Φ*w is:
    p(f* | Φ*, y, Φ, α, β) = N(f* | μ_f, Σ_f)
    
    Where:
    μ_f = Φ* m                              (predictive mean)
    Σ_f = Φ* S Φ*^T                        (predictive covariance)
    
    For output predictions y* = f* + ε:
    p(y* | Φ*, y, Φ, α, β) = N(y* | μ_y, Σ_y)
    
    Where:
    μ_y = μ_f                               (same mean)
    Σ_y = Σ_f + β^(-1)I                    (additional noise variance)
    
    Parameters:
    -----------
    Phi : jnp.ndarray, shape (N, D)
        Design matrix where N is number of data points and D is number of features
    y : jnp.ndarray, shape (N, 1)
        Target values
    alpha : float, default=1.0
        Precision (inverse variance) of the weight prior: p(w) ~ N(0, α^(-1)I)
        Higher α means stronger regularization (weights closer to zero)
    beta : float, default=1.0
        Precision (inverse variance) of the noise: p(ε) ~ N(0, β^(-1))
        Higher β means lower noise assumption
    
    Attributes:
    -----------
    Phi : jnp.ndarray
        Stored design matrix
    y : jnp.ndarray
        Stored target values
    N : int
        Number of data points
    D : int
        Number of features/basis functions
    alpha : float
        Weight precision hyperparameter
    beta : float
        Noise precision hyperparameter
    m : jnp.ndarray, shape (D, 1)
        Posterior mean of weights
    S : jnp.ndarray, shape (D, D)
        Posterior covariance matrix of weights
    log_marginal_likelihood : float
        Log marginal likelihood p(y|α,β)
    
    Example:
    --------
    >>> import jax.numpy as jnp
    >>> # Create synthetic data
    >>> N, D = 100, 3
    >>> Phi = jnp.random.normal(0, 1, (N, D))  # Design matrix
    >>> true_weights = jnp.array([1.5, -0.8, 2.1])
    >>> y = (Phi @ true_weights + 0.1 * jnp.random.normal(0, 1, N)).reshape(-1, 1)
    >>> 
    >>> # Fit Bayesian linear regression
    >>> model = BayesianLinearRegression(Phi, y, alpha=1.0, beta=25.0)
    >>> 
    >>> # Make predictions on new data
    >>> Phi_test = jnp.random.normal(0, 1, (10, D))
    >>> mu_y, var_y = model.predict_y(Phi_test)
    >>> print(f"Predictive mean: {mu_y}")
    >>> print(f"Predictive variance: {var_y}")
    """
    
    def __init__(self, Phi, y, alpha=1., beta=1.):
        """
        Initialize Bayesian Linear Regression model.
        
        Computes the posterior distribution p(w|y,Φ,α,β) = N(w|m,S) analytically
        and stores all relevant quantities for prediction and inference.
        
        Parameters:
        -----------
        Phi : jnp.ndarray, shape (N, D)
            Design matrix containing basis function evaluations
        y : jnp.ndarray, shape (N, 1)
            Target values (must be column vector)
        alpha : float, default=1.0
            Precision of weight prior (higher = more regularization)
        beta : float, default=1.0
            Precision of noise (higher = assume less noise)
        
        Example:
        --------
        >>> Phi = jnp.array([[1, 2], [1, 3], [1, 4]])  # Linear basis [1, x]
        >>> y = jnp.array([[3], [4], [5]])              # Linear relationship
        >>> model = BayesianLinearRegression(Phi, y, alpha=1.0, beta=1.0)
        """
        
        # store data and hyperparameters
        self.Phi, self.y = Phi, y
        self.N, self.D = Phi.shape
        self.alpha, self.beta = alpha, beta
        
        # compute posterior distribution
        self.m, self.S = self.compute_posterior(alpha, beta)
        self.log_marginal_likelihood = self.compute_marginal_likelihood(alpha, beta)

        # perform sanity check of shapes/dimensions
        self.check_dimensions()
        
    def check_dimensions(self):
        """
        Verify that all arrays have correct dimensions.
        
        This is a sanity check to ensure mathematical operations are valid
        and catch dimension mismatches early in development.
        
        Raises:
        -------
        AssertionError
            If any array has incorrect dimensions
        
        Expected Dimensions:
        -------------------
        y : (N, 1) - targets must be column vector
        m : (D, 1) - posterior mean must be column vector  
        S : (D, D) - posterior covariance must be square matrix
        
        Example:
        --------
        >>> model = BayesianLinearRegression(Phi, y)
        >>> model.check_dimensions()  # Will raise error if shapes are wrong
        """
        D = self.D
        N = self.N
        
        # Check target vector dimensions
        assert self.y.shape == (N, 1), (
            f"Wrong shape for data vector y.\n"
            f"For N = {N}, the shape of y must be ({N}, 1), "
            f"but the actual shape is {self.y.shape}"
        )
        
        # Check posterior mean dimensions  
        assert self.m.shape == (D, 1), (
            f"Wrong shape for posterior mean.\n"
            f"For D = {D}, the shape of the posterior mean must be ({D}, 1), "
            f"but the actual shape is {self.m.shape}"
        )
        
        # Check posterior covariance dimensions
        assert self.S.shape == (D, D), (
            f"Wrong shape for posterior covariance.\n"
            f"For D = {D}, the shape of the posterior covariance must be ({D}, {D}), "
            f"but the actual shape is {self.S.shape}"
        )
    def compute_posterior(self, alpha, beta):
        """
        Compute the posterior distribution p(w|y,Φ,α,β) = N(w|m,S) analytically.
        
        This implements the core Bayesian inference for linear regression.
        Given a Gaussian prior p(w) = N(0, α^(-1)I) and Gaussian likelihood
        p(y|w,Φ,β) = N(Φw, β^(-1)I), the posterior is also Gaussian.
        
        Mathematical Derivation:
        ========================
        Starting from Bayes' theorem:
        p(w|y,Φ,α,β) ∝ p(y|w,Φ,β) p(w|α)
        
        Both likelihood and prior are Gaussian, so posterior is Gaussian:
        p(w|y,Φ,α,β) = N(w|m,S)
        
        The posterior precision (inverse covariance) combines prior and data precision:
        S^(-1) = α I + β Φ^T Φ
        
        Where:
        - α I represents prior precision (regularization)
        - β Φ^T Φ represents data precision (how much data informs each parameter)
        
        The posterior covariance is:
        S = (α I + β Φ^T Φ)^(-1)
        
        The posterior mean balances prior (zero) and data evidence:
        m = β S Φ^T y
        
        Intuition:
        - If α is large: strong prior → m closer to 0, S smaller (more certain)
        - If β is large: trust data more → m closer to least squares solution
        - If data is abundant: Φ^T Φ large → posterior dominated by data
        
        Parameters:
        -----------
        alpha : float
            Prior precision parameter
        beta : float  
            Noise precision parameter
            
        Returns:
        --------
        m : jnp.ndarray, shape (D, 1)
            Posterior mean vector
        S : jnp.ndarray, shape (D, D)
            Posterior covariance matrix
            
        Example:
        --------
        >>> model = BayesianLinearRegression(Phi, y)
        >>> m, S = model.compute_posterior(alpha=2.0, beta=1.0)
        >>> print(f"Posterior mean: {m.ravel()}")
        >>> print(f"Posterior uncertainty (diagonal of S): {jnp.diag(S)}")
        """
        #################################alpha############
        # Bayesian Linear Regression Posterior Computation
        #############################################
        
        # Prior precision matrix: p(w) = N(0, α^(-1)I)
        # This represents our belief that weights should be close to zero
        # Higher α = stronger regularization = weights pulled toward zero
        S0_inv = alpha * jnp.eye(self.D)  # Prior precision: α I
        
        # Posterior precision matrix (inverse covariance):
        # S^(-1) = α I + β Φ^T Φ
        # This combines:
        # - Prior precision α I (regularization term)
        # - Data precision β Φ^T Φ (how much each parameter is constrained by data)
        A = S0_inv + beta * self.Phi.T @ self.Phi  # Posterior precision matrix
        
        # Posterior covariance matrix:
        # S = (α I + β Φ^T Φ)^(-1)
        # This tells us the uncertainty in our weight estimates
        # Smaller values = more certain estimates
        S = jnp.linalg.inv(A)  # Posterior covariance matrix
        
        # Posterior mean:
        # m = β S Φ^T y
        # This is the weighted combination of prior mean (0) and data evidence
        # When β is large (low noise), this approaches the least squares solution
        m = beta * S @ self.Phi.T @ self.y  # Posterior mean vector
        
        #############################################
        # End of solution
        #############################################
        return m, S      
    def generate_prior_samples(self, key, num_samples):
        """ generate samples from the prior  """
        return random.multivariate_normal(key, jnp.zeros(len(self.m)), (1/self.alpha)*jnp.identity(len(self.m)), shape=(num_samples, ))
    
    def generate_posterior_samples(self, key, num_samples):
        """ generate samples from the posterior  """
        return random.multivariate_normal(key, self.m.ravel(), self.S, shape=(num_samples, ))
    
    def predict_f(self, Phi):
        """ computes posterior mean (mu_f) and variance (var_f) of f(phi(x)) for each row in Phi-matrix. If Phi is a [N, D]-matrix, then the shapes of both mu_f and var_f must be (N,). The function returns (mu_f, var_f)
        """
        mu_f = (Phi@self.m).ravel()   
        var_f = jnp.diag(Phi@self.S@Phi.T)   
        
        # check dimensions before returning values
        assert mu_f.shape == (Phi.shape[0],), "Shape of mu_f seems wrong. Check your implementation"
        assert var_f.shape == (Phi.shape[0],), "Shape of var_f seems wrong. Check your implementation"
        return mu_f, var_f
        
    def predict_y(self, Phi):
        """ returns posterior predictive mean (mu_y) and variance (var_y) of y = f(phi(x)) + e for each row in Phi-matrix. If Phi is a [N, D]-matrix, then the shapes of both mu_y and var_y must be (N,). The function returns (mu_y, var_y)
        """
        mu_f, var_f = self.predict_f(Phi)
        mu_y = mu_f                  
        var_y = var_f + 1/self.beta  

        # check dimensions before returning values
        assert mu_y.shape == (Phi.shape[0],), "Shape of mu_y seems wrong. Check your implementation"
        assert var_y.shape == (Phi.shape[0],), "Shape of var_y seems wrong. Check your implementation"
        return mu_y, var_y
        
    
    def compute_marginal_likelihood(self, alpha, beta):
        """ computes and returns log marginal likelihood p(y|alpha, beta) """
        inv_S0 = alpha*jnp.identity(self.D)
        A = inv_S0 + beta*(self.Phi.T@self.Phi)
        m = beta*jnp.linalg.solve(A, self.Phi.T)@self.y   # (eq. 3.53 in Bishop)
        S = jnp.linalg.inv(A)                             # (eq. 3.54 in Bishop)
        Em = beta/2*jnp.sum((self.y - self.Phi@m)**2) + alpha/2*jnp.sum(m**2)
        return self.D/2*jnp.log(alpha) + self.N/2*jnp.log(beta) - Em - 0.5*jnp.linalg.slogdet(A)[1] - self.N/2*jnp.log(2*jnp.pi)
         

    def optimize_hyperparameters(self):
        # optimizes hyperparameters using marginal likelihood
        theta0 = jnp.array((jnp.log(self.alpha), jnp.log(self.beta)))
        def negative_marginal_likelihood(theta):
            alpha, beta = jnp.exp(theta[0]), jnp.exp(theta[1])
            return -self.compute_marginal_likelihood(alpha, beta)

        result = minimize(value_and_grad(negative_marginal_likelihood), theta0, jac=True)

        # store new hyperparameters and recompute posterior
        theta_opt = result.x
        self.alpha, self.beta = jnp.exp(theta_opt[0]), jnp.exp(theta_opt[1])
        self.m, self.S = self.compute_posterior(self.alpha, self.beta)
        self.log_marginal_likelihood = self.compute_marginal_likelihood(self.alpha, self.beta)


    def get_w_mle(self):
        """
        Compute the Maximum Likelihood Estimate (MLE) of weights.
        
        This corresponds to the least squares solution:
        w_MLE = (ΦᵀΦ)^(-1) Φᵀy
        
        Returns:
        --------
        w_mle : jnp.ndarray, shape (D, 1)
            MLE estimate of weights
        """
        PhiT_Phi_inv = jnp.linalg.inv(self.Phi.T @ self.Phi)
        w_mle = PhiT_Phi_inv @ self.Phi.T @ self.y
        return w_mle

    def get_sigma2_mle(self):
        """
        Compute the MLE of the noise variance σ² using the residuals of the MLE fit:
        σ²_MLE = (1/N) * ||y - Φ w_MLE||²

        Returns:
        --------
        sigma2_mle : float
            Maximum likelihood estimate of the noise variance
        """
        w_mle = self.get_w_mle()
        residuals = self.y - self.Phi @ w_mle
        sigma2_mle = jnp.sum(residuals**2) / self.N
        return sigma2_mle

    def get_w_map(self):
        """
        Returns the MAP estimate of weights.

        Since the prior and likelihood are Gaussian, the posterior is Gaussian,
        and the MAP estimate is simply the posterior mean `m`.

        Returns:
        --------
        w_map : jnp.ndarray, shape (D, 1)
            MAP estimate of weights (posterior mean)
        """
        return self.m
    
    
"""
Bayesian Linear Regression Visualization with Progressive Data Updates

This script demonstrates how Bayesian linear regression evolves as more training data
is observed. It shows the progression from prior beliefs through likelihood updates
to posterior distributions, illustrating the fundamental principles of Bayesian learning.

Mathematical Foundation:
- Prior: p(w) = N(w | 0, α⁻¹I) where α controls prior precision
- Likelihood: p(y|w,X) = N(y | Φw, β⁻¹I) where β controls noise precision  
- Posterior: p(w|y,X) = N(w | μₙ, Σₙ) (analytically tractable for linear models)

Key Equations:
- Posterior mean: μₙ = βΣₙΦᵀy
- Posterior covariance: Σₙ = (αI + βΦᵀΦ)⁻¹
- Predictive mean: μ(x*) = μₙᵀφ(x*)
- Predictive variance: σ²(x*) = φ(x*)ᵀΣₙφ(x*) + β⁻¹
"""

def plot_predictions(ax, x, mu, var, color='r', visibility=0.5, label=None):
    """
    Visualize predictive distributions with mean and confidence intervals.
    
    This function plots the predictive mean along with 95% confidence intervals,
    providing a visual representation of model uncertainty.
    
    Mathematical basis:
    - Confidence interval: μ ± 1.96σ (covers ~95% of probability mass)
    - For Gaussian distributions: P(μ - 1.96σ ≤ X ≤ μ + 1.96σ) ≈ 0.95
    
    Args:
        ax: Matplotlib axis object for plotting
        x: Input locations where predictions are made
           Shape: (n_pred, 1) - prediction points
        mu: Predictive mean at each input location  
            Shape: (n_pred, 1) - E[f(x*)] or E[y*]
        var: Predictive variance at each input location
             Shape: (n_pred, 1) - Var[f(x*)] or Var[y*]
        color: Color for the prediction plots
        visibility: Transparency level for confidence bands (0-1)
        label: Legend label for the prediction
    
    Visual elements:
    - Solid line: Predictive mean μ(x)
    - Dashed lines: Confidence bounds μ(x) ± 1.96σ(x)
    - Shaded area: Uncertainty region between bounds
    """
    # Calculate 95% confidence bounds using standard normal quantiles
    # 1.96 is the 97.5th percentile of standard normal (two-tailed 95% CI)
    lower = mu - 1.96 * jnp.sqrt(var)  # Lower confidence bound
    upper = mu + 1.96 * jnp.sqrt(var)  # Upper confidence bound
    
    # Plot predictive mean as solid line
    ax.plot(x, mu, color=color, label=label)
    
    # Plot confidence bounds as dashed lines
    ax.plot(x, lower, color=color, linewidth=2, linestyle='--')
    ax.plot(x, upper, color=color, linewidth=2, linestyle='--')
    
    # Fill area between bounds to show uncertainty region
    ax.fill_between(x.ravel(), lower.ravel(), upper.ravel(), 
                   color=color, alpha=visibility)
    
    # Emphasize the mean with a thicker line
    ax.plot(x, mu, '-', color=color, label="", linewidth=2.5)
