from scipy.stats import norm
import jax.numpy as jnp
from scipy.optimize import minimize

sigmoid = lambda x: 1./(1+jnp.exp(-x))
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import seaborn as snb

from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import minimize
from jax import value_and_grad
from jax import grad
from jax.scipy.stats import norm
from jax import hessian
# from jax.misc.optimizers import adam
from jax.flatten_util import ravel_pytree

def log_npdf(x, m, v):
    return -0.5*(x-m)**2/v - 0.5*jnp.log(2*jnp.pi*v)




#######################################################################################
# Neural network model with MAP inference
# Adapted from: # https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
########################################################################################

class NeuralNetworkMAP(object):

    def __init__(self, X, y, layer_sizes, likelihood, alpha=1., step_size=0.01, max_itt=1000, seed=0):

        # data
        self.X = X
        self.y = y

        # model and optimization parameters
        self.likelihood = likelihood(y)
        self.layer_sizes = layer_sizes
        self.step_size = step_size
        self.max_itt = max_itt
        self.alpha = alpha

        # random number genration
        self.seed = seed
        self.key = random.PRNGKey(self.seed)
        
        # initialize parameters and optimize
        self.params = self.init_random_params()
        self.optimize_adam()


    def init_random_params(self):
        """Build a list of (weights, biases) tuples,
        one for each layer in the net."""
        parameters = []
        for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.key, subkey = random.split(self.key)
            w_key, b_key = random.split(subkey, 2)
            weight_matrix =jnp.sqrt(2/n) * random.normal(w_key, shape=(m, n))
            bias_vector = jnp.sqrt(2/n) * random.normal(b_key, shape=(n))
            parameters.append((weight_matrix, bias_vector))

        return parameters

    def neural_net_predict(self, params, inputs):
        """Implements a deep neural network for classification.
        params is a list of (weights, bias) tuples.
        inputs is an (N x D) matrix.
        returns logits."""
        for W, b in params:
            outputs = jnp.dot(inputs, W) + b
            inputs = jnp.tanh(outputs)
        return outputs# - logsumexp(outputs, axis=1, keepdims=True)

    def predict(self, inputs):
        return self.neural_net_predict(self.params, inputs)

    def log_prior(self, params):
        # implement a Gaussian prior on the weights
        flattened_params, _ = ravel_pytree(params)
        return  jnp.sum(log_npdf(flattened_params, 0., 1/self.alpha))
    
    def log_likelihood(self, params):     
        y = self.neural_net_predict(params, self.X)
        return self.likelihood.log_lik(y.ravel())

    def log_posterior(self, params):
        return self.log_prior(params) + self.log_likelihood(params)

    def optimize_adam(self, b1=0.9, b2=0.999, eps=1e-8):
    
        # Define training objective and gradient of objective using autograd.
        def objective(params, iter):
            return -self.log_posterior(params)
            
        objective_grad = grad(objective)
        params_flat, unflatten = ravel_pytree(self.params)
        params = self.params

        m = jnp.zeros(len(params_flat))
        v = jnp.zeros(len(params_flat))

        for itt in range(self.max_itt):

            # compute gradient and flatten
            g = objective_grad(params, itt)
            g, _ = ravel_pytree(g)

            # ADAM update rules
            m = (1 - b1) * g + b1 * m  # First  moment estimate.
            v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
            mhat = m / (1 - b1 ** (itt + 1))  # Bias correction.
            vhat = v / (1 - b2 ** (itt + 1))
            params_flat = params_flat - self.step_size * mhat / (jnp.sqrt(vhat) + eps)
            params = unflatten(params_flat)

        self.params = params

        return self

#######################################################################################
# Helper function for sampling multivariate Gaussians
########################################################################################


def generate_samples(key, mean, K, M, jitter=1e-8):
    """ returns M samples from a zero-mean Gaussian process with kernel matrix K
    
    arguments:
    K      -- NxN kernel matrix
    M      -- number of samples (scalar)
    jitter -- scalar
    returns NxM matrix
    """
    
    L = jnp.linalg.cholesky(K + jitter*jnp.identity(len(K)))
    zs = random.normal(key, shape=(len(K), M))
    fs = mean + jnp.dot(L, zs)
    return fs


#######################################################################################
# Kernels
########################################################################################

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
            squared_exponential = lambda tau, kappa, lengthscale: kappa**2*jnp.exp(-0.5*tau**2/lengthscale**2)
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

        # compute all the pairwise distances efficiently
        dists = jnp.sqrt(jnp.sum((jnp.expand_dims(X1, 1) - jnp.expand_dims(X2, 0))**2, axis=-1))
        
        # squared exponential covariance function
        K = self.kernel_fun(dists, kappa, lengthscale)
        
        # add jitter to diagonal for numerical stability
        if len(X1) == len(X2) and jnp.allclose(X1, X2):
            K = K + jitter*jnp.identity(len(X1))
                
        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but the actual shape was {K.shape}. Please check your code. "
        return K




#######################################################################################
# For plotting
########################################################################################

def plot_with_uncertainty(ax, Xp, mu, Sigma, sigma=0, color='g', color_samples='g', title="", num_samples=0, seed=0):
    
    mean, std = mu.ravel(), jnp.sqrt(jnp.diag(Sigma) + sigma**2)

    
    # plot distribution
    ax.plot(Xp, mean, color=color, label='GP')
    ax.plot(Xp, mean + 2*std, color=color, linestyle='--')
    ax.plot(Xp, mean - 2*std, color=color, linestyle='--')
    ax.fill_between(Xp.ravel(), mean - 2*std, mean + 2*std, color=color, alpha=0.25, label='95% interval')
    
    # generate samples
    if num_samples > 0:
        key = random.PRNGKey(seed)
        fs = generate_samples(key, mu[:, None], Sigma, num_samples, 1e-6)
        ax.plot(Xp, fs, color=color_samples, alpha=.25)
    
    ax.set_title(title)

    if num_samples > 0:
        return fs
    

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
        
def eval_density_grid(density_fun, P=100, a=-5, b=5):
    x_grid = jnp.linspace(a, b, P)
    X1, X2 = jnp.meshgrid(x_grid, x_grid)
    XX = jnp.column_stack((X1.ravel(), X2.ravel()))
    return x_grid, density_fun(XX).reshape((P, P))


#######################################################################################
# compute classification error and std. error of the mean
########################################################################################


def compute_err(t, tpred):
    return jnp.mean(tpred.ravel() != t), jnp.std(tpred.ravel() != t)/jnp.sqrt(len(t))

#######################################################################################
# load subset of mnist data
########################################################################################

def load_MNIST_subset(filename, digits=[4,7], plot=True, subset=300, seed=0):

    data = jnp.load(filename)
    images = data['images']
    labels = data['labels']

    # we will only focus on binary classification using two digits
    idx = jnp.logical_or(labels == digits[0], labels == digits[1])
    
    # extract digits of interest
    X = images[idx, :]
    t = labels[idx].astype('float')
    
    # set labels to 0/1/2/...
    for i in range(len(digits)):
        t[t == digits[i]] = i
        
    # split into training/test
    N = len(X)
    Ntrain = int(0.5*N)
    Ntest = N - Ntrain
    key = random.PRNGKey(seed)
    train_idx = random.choice(key, jnp.arange(N), shape=(Ntrain,), replace=False)
    test_idx = jnp.setdiff1d(jnp.arange(N), train_idx)

    Xtrain = X[train_idx, :]
    Xtest = X[test_idx, :]
    ttrain = t[train_idx]
    ttest = t[test_idx]

    # standardize training set
    Xm = Xtrain.mean(0)
    Xs = Xtrain.std(0)
    Xs[Xs == 0] = 1 # avoid division by zero for "always black" pixels

    Xtrain_std = (Xtrain - Xm)/Xs
    Xtest_std = (Xtest - Xm)/Xs


    # reduce dimensionality to 2D using principal component analysis (PCA)
    U, s, V = jnp.linalg.svd(Xtrain_std)

    # get eigenvectors corresponding to the two largest eigenvalues
    eigen_vecs = V[:2, :]
    eigen_vals = s[:2]

    # set-up projection matrix
    Pmat = eigen_vecs.T*(jnp.sqrt(Ntrain)/eigen_vals)

    # project and standize
    Phi_train =Xtrain_std@Pmat
    Phi_test = Xtest_std@Pmat

    # Let's only use a small subset of the training data
    Phi_train = Phi_train[:subset]
    ttrain = ttrain[:subset]

    return Phi_train, Phi_test, ttrain, ttest,


probit = lambda x: norm.cdf(x)

class GaussianProcessClassification(object):

    def __init__(self, X, y, likelihood, kernel, kappa=1., lengthscale=1.,jitter=1e-8):
        """  
        Arguments:
            X                -- NxD input points
            y                -- Nx1 observed values 
            likelihood       -- likelihood instance
            kernel           -- must be instance of the StationaryIsotropicKernel class
            jitter           -- non-negative scaler
            kappa            -- magnitude (positive scalar)
            lengthscale      -- characteristic lengthscale (positive scalar)
        """
        self.X = X
        self.y = y
        self.N = len(X)
        self.likelihood = likelihood(y)
        self.kernel = kernel
        self.jitter = jitter
        self.set_hyperparameters(kappa, lengthscale)

        # precompute kernel, its Cholesky decomposition and prepare Laplace approx
        self.K = self.kernel.contruct_kernel(self.X, self.X, jitter=self.jitter)
        self.L = jnp.linalg.cholesky(self.K)
        self.construct_laplace_approximation()

    def set_hyperparameters(self, kappa, lengthscale):
        self.kernel.kappa = kappa
        self.kernel.lengthscale = lengthscale
        
    def log_joint_a(self, a):
        """ computes and returns the log joint distribution log p(y, f), where f = K*a """
        f = self.K@a
        # compute log prior contribution
        const = -self.N/2*jnp.log(2*jnp.pi)
        logdet = jnp.sum(jnp.log(jnp.diag(self.L)))
        quad_term =  0.5*jnp.sum(a*f)
        log_prior = const - logdet - quad_term
        # compute log likelihood contribution
        log_lik = self.likelihood.log_lik(f)
        # return sum
        return log_prior + log_lik
    

    def grad_a(self, a):
        """ computes gradient of log joint distribution, i.e. log p(y, a) = log p(y|a) + log p(a), wrt. a """
        f = self.K@a
        # compute gradient contribution from prior and likelihood
        grad_prior = -f
        grad_lik = self.likelihood.grad(f)@self.K
        # sum and return
        return grad_prior + grad_lik
        
    
    def compute_f_MAP(self):
        # optimize to get f_MAP
        result = minimize(lambda a: -self.log_joint_a(a), jac=lambda a: -self.grad_a(a), x0=jnp.zeros((self.N)))
        
        if not result.success:
            print(result)
            raise ValueError('Optization failed')
        
        self.a = result.x
        f_MAP = self.K @ result.x
        return f_MAP

    def construct_laplace_approximation(self):

        # f_MAP
        self.m = self.compute_f_MAP()


        Lambda = -self.likelihood.hessian(self.m)
        # Compute Hessian
        self.H = -Lambda - jnp.linalg.inv(self.K)
        self.S = jnp.linalg.inv(-self.H)

    def predict_f(self, Xstar):
        """ returns the posterior distribution of f^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Arguments:
        Xstar            -- PxD prediction points
        
        returns:
        mu               -- mean vector, shape (P,)
        Sigma            -- covariance matrix, shape (P, P) 
        """
        ##############################################
        # Your solution goes here
        ##############################################
        # Covariances 
        K = self.K              # (N, N), from training 
        m, S = self.m, self.S   # (N,),  (N, N)
        
        K_star_x = self.kernel.contruct_kernel(X1=Xstar, X2=self.X, jitter=self.jitter)       # (P, N)
        K_star_star = self.kernel.contruct_kernel(X1=Xstar, X2=Xstar, jitter=self.jitter)     # (P, P)
        
        # Stable solvers (not explicit inverse)
        K_inv_m = jnp.linalg.solve(K, m)
        H = jnp.linalg.solve(K, K_star_x.T)
        
        
        mu = K_star_x @ K_inv_m
        Sigma = K_star_star - H.T @ (self.K - S) @ H
        ##############################################
        # End of solution
        ##############################################

        # check dimensions and return
        assert (mu.shape == (len(Xstar),)), f"Expected shape for mu is ({len(Xstar)}), but the actual shape was {mu.shape}. Please check implementation"
        assert Sigma.shape == (len(Xstar), len(Xstar)), f"Expected shape for Sigma is ({len(Xstar)}, {len(Xstar)}), but the actual shape was {Sigma.shape}. Please check implementation"

        return mu, Sigma
    
    def predict_y(self, Xstar):
        """ returns the posterior distribution of y^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Arguments:
        Xstar            -- PxD prediction points
        
        returns:
        p               -- vector of post. pred. probabilities, shape (P,)
        """
        ##############################################
        # Your solution goes here
        ##############################################
        mu_f, sigma_f = self.predict_f(Xstar)        
        p = probit(mu_f / (jnp.sqrt( 8 / jnp.pi  + jnp.diag(sigma_f))))
        ##############################################
        # End of solution
        ##############################################

        # check dimensions and return
        assert (p.shape == (len(Xstar),)), f"Expected shape for p is ({len(Xstar)}), but the actual shape was {p.shape}. Please check implementation"
        return p
    
    def posterior_samples(self, Xstar, num_samples):
        """
            generate samples from the posterior p(f^*|y, x^*) for each of the inputs in Xstar

            Arguments:
                Xstar            -- PxD prediction points
        
            returns:
                f_samples        -- numpy array of (P, num_samples) containing num_samples for each of the P inputs in Xstar
        """
        mu, Sigma = self.predict_f(Xstar)
        f_samples = generate_samples(mu.ravel(), Sigma, num_samples)

        assert (f_samples.shape == (len(Xstar), num_samples)), f"The shape of the posterior mu seems wrong. Expected ({len(Xstar)}, {num_samples}), but actual shape was {f_samples.shape}. Please check implementation"
        return f_samples
    
    def prior_predict_f(self, Xstar):
        """
        Prior predictive for f* at Xstar (no data used).
        Returns:
            mu0   : (P,)   zero vector
            Sigma0: (P,P)  K(X*, X*)
        """
        K_star_star = self.kernel.contruct_kernel(X1=Xstar, X2=Xstar, jitter=self.jitter)  # (P,P)
        mu0 = jnp.zeros((len(Xstar),), dtype=K_star_star.dtype)
        Sigma0 = K_star_star
        # sanity checks
        assert mu0.shape == (len(Xstar),), f"mu0 shape {mu0.shape} != ({len(Xstar)},)"
        assert Sigma0.shape == (len(Xstar), len(Xstar)), \
            f"Sigma0 shape {Sigma0.shape} != ({len(Xstar)}, {len(Xstar)})"
        return mu0, Sigma0

    def prior_predict_y(self, Xstar):
        """
        Prior predictive for y* at Xstar under a probit likelihood:
            p(y*=1 | X*) = E_f*[Phi(f*)] = Phi( mu0 / sqrt(1 + var0) )
        For the zero-mean GP prior, this simplifies to 0.5 everywhere.
        Returns:
            p: (P,) prior predictive probabilities.
        """
        mu0, Sigma0 = self.prior_predict_f(Xstar)
        var0 = jnp.clip(jnp.diag(Sigma0), a_min=0.0)
        p = probit(mu0 / jnp.sqrt(1.0 + var0))  # equals 0.5 when mu0=0
        assert p.shape == (len(Xstar),), f"p shape {p.shape} != ({len(Xstar)},)"
        return p

    def prior_samples(self, Xstar, num_samples):
        """
        Draw samples of f* ~ N(0, K(X*,X*)) from the GP prior.
        Returns:
            f_samples: (P, num_samples)
        """
        mu0, Sigma0 = self.prior_predict_f(Xstar)
        f_samples = generate_samples(mu0.ravel(), Sigma0, num_samples)
        assert f_samples.shape == (len(Xstar), num_samples)
        return f_samples


from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.special import logsumexp, expit, logit

class BernoulliLikelihood(object):
    """ Implement the Bernoulli likelihood with the sigmoid as inverse link function """

    def __init__(self, y):
        # store data & force shape (N, )
        self.y = y.ravel()

    def log_lik(self, f):
        """ Implements log p(y|f) = sum log p(y_n|f_n), where p(y_n|f_n) = Ber(y_n|sigmoid(f_n)). 
            
            Argument:
            f       --       vector of function values, shape (N, )

            Returns
            ll      --       sum of log likelihoods for all N data points, scalar

        """
        ##############################################
        # Your solution goes here
        ##############################################
        p = sigmoid(f)
        ll = jnp.sum( 
                        self.y * jnp.log( p ) + ( 1 - self.y ) * jnp.log( 1 - p )
                    )
        ##############################################
        # End of solution
        ##############################################

        # check shape and return
        assert ll.shape == (), f"Expected shape for loglik_ is (), but the actual shape was {ll.shape}. Please check implementation"
        return ll
    
    def grad(self, f):
        """ Implements the gradient of log p(y|n) 

            Argument:
            f       --       vector of function values, shape (N, )

            Returns
            g       --       gradient of log p(y|f), i.e. a vector of first order derivatives with shape (N, )
             
        """
        ##############################################
        # Your solution goes here
        ##############################################

        g = self.y - sigmoid(f)
        
        ##############################################
        # End of solution
        ##############################################
        # check shape and return
        assert g.shape == (len(f), ), f"Expected shape for g is ({len(f)}, ), but the actual shape was {g.shape}. Please check implementation"
        return g

    def hessian(self, f):
        """ Implements the Hessian of log p(y|n) 

        Argument:
            f       --       vector of function values, shape (N, )

        Returns:
            Lambda  --       Hessian of likelihood, i.e. a diagonal matrix with the second order derivatives on the diagonal, shape (N, N)
        """

        ##############################################
        # Your solution goes here
        ##############################################
        p = sigmoid(f)
        Lambda = jnp.diag(- p * (1 - p))
        ##############################################
        # End of solution
        ##############################################

        # check shape and return
        assert Lambda.shape == (len(f), len(f)), f"Expected shape for Lambda is ({len(f)}, {len(f)}), but the actual shape was {Lambda.shape}. Please check implementation"
        return Lambda

