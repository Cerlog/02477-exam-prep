import jax.numpy as jnp

def entropy(p, mine=True):
    """
    Computes the entropy of the class probability distribution at each prediction point.

    Entropy is a measure of uncertainty:
    - High entropy → flat distribution → model is uncertain
    - Low entropy → peaked distribution → model is confident

    Mathematically:
        entropy_n = -Σ_k p[n, k] * log(p[n, k])

    Args:
        p: jnp.ndarray of shape (N, K)
           - Softmax probabilities per prediction point
           - Should sum to 1 across axis=1

    Returns:
        jnp.ndarray of shape (N,)
            - Entropy values for each prediction point

    Why `axis=1`?
        Because we sum over the **class probabilities** for each input point.

    Example:
        >>> p = jnp.array([[0.25, 0.25, 0.25, 0.25],
        ...                [0.97, 0.01, 0.01, 0.01]])
        >>> entropy(p)
        Array([1.3863, 0.1669], dtype=float32)  # Higher = more uncertain
    """
    if mine:
        return -jnp.sum(p * jnp.log(p), axis=1)
    else: 
        idx = p > 0
        return -jnp.sum(p[idx] * jnp.log(p[idx]))

def confidence(phat):
    """
    Computes the confidence score by selecting the highest probability in each row
    using argmax and advanced indexing.

    This is equivalent to confidence_teacher, but explicitly demonstrates how to
    extract the value at the predicted class index.

    Mathematically:
        k_star[n] = argmax_k phat[n, k]
        confidence[n] = phat[n, k_star[n]]

    Args:
        phat: jnp.ndarray of shape (N, K)
              - Softmax probabilities for N prediction points across K classes

    Returns:
        jnp.ndarray of shape (N,)
            - Confidence scores: maximum predicted probability per row

    Example:
        >>> phat = jnp.array([[0.1, 0.7, 0.2],
        ...                   [0.3, 0.4, 0.3]])
        >>> confidence(phat)
        Array([0.7, 0.4], dtype=float32)
    """
    phat = jnp.asarray(phat)
    was_1d = (phat.ndim == 1)
    phat2 = jnp.atleast_2d(phat)            # ensure shape (N, K)
    
    idx = jnp.argmax(phat2, axis=1)         # (N,)
    row_indices = jnp.arange(phat2.shape[0])
    vals = phat2[row_indices, idx]          # (N,)
    
    return vals[0] if was_1d else vals


def compute_expected_utility(U, phat):
    """ 
    Computes the expected utility for a multi-class classification problem 
    with K classes for utility matrix U and posterior predictive probabilities phat.
    
    Arguments
    ---------
    U : np.ndarray, shape (K, K)
        Utility matrix, where U[i, j] = utility of predicting class j when the true class is i.
    phat : np.ndarray, shape (P, K)
        Posterior predictive probabilities for P points over K classes.

    Returns
    -------
    expected_util : np.ndarray, shape (P, K)
        Expected utility for each prediction point (rows) and each possible predicted class (columns).

    Formula
    -------
    For each prediction point p and each possible predicted class j:
        expected_util[p, j] = sum_{true_class=i} phat[p, i] * U[i, j]
    """

    expected_util = phat @ U  # (P, K) @ (K, K) → (P, K)

    # Shape check
    assert expected_util.shape == phat.shape, \
        f'Expected {phat.shape}, but got {expected_util.shape}'

    return expected_util


# ===========================
# EXAMPLE USAGE (commented out)
# ===========================
#
# # Example data for testing
# phat = jnp.array([
#     [0.25, 0.25, 0.25, 0.25],  # uniform probs → high uncertainty
#     [0.97, 0.01, 0.01, 0.01]   # peaked probs → low uncertainty
# ])
#
# # 1) Entropy
# ent = entropy(phat)
# print("Entropy:", ent)
#
# # 2) Confidence
# conf = confidence(phat)
# print("Confidence:", conf)
#
# # 3) Expected Utility
# U = jnp.array([
#     [1.0, 0.0, -1.0, 0.5],
#     [0.0, 1.0,  0.5, 0.0],
#     [-1.0, 0.5, 1.0, 0.0],
#     [0.5, 0.0,  0.0, 1.0]
# ])
#
# expected_util = compute_expected_utility(U, phat)
# print("Expected Utility:\n", expected_util)
