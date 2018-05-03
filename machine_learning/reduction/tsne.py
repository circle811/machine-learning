import numpy as np

from ..algorithm.optimizer import MomentumGD
from ..utils.distance import pairwise_l2_distance

__all__ = ['TSNE']

EPS = np.finfo(np.float64).eps
TINY = np.finfo(np.float64).tiny


def compute_perplexity(d, gamma):
    # probability
    c = np.exp(-gamma * d)
    p = c / np.maximum(TINY, np.sum(c))

    # entropy
    lp = np.log(p, out=np.zeros_like(p), where=p > 0)
    h = -np.sum(p * lp)

    return np.exp(h), p


def binary_search_gamma(d, desired_perplexity, max_iter, tol):
    gamma = 1.0
    gamma_lower = 0.0
    gamma_upper = np.inf
    for i in range(max_iter):
        perplexity, p = compute_perplexity(d, gamma)
        if perplexity < desired_perplexity - tol:
            gamma_upper = gamma
            gamma = (gamma_lower + gamma_upper) / 2
        elif perplexity > desired_perplexity + tol:
            gamma_lower = gamma
            if gamma_upper == np.inf:
                gamma = gamma_lower * 2
            else:
                gamma = (gamma_lower + gamma_upper) / 2
        else:
            break
    return gamma, perplexity, p


def conditional_probability(X, desired_perplexity, max_iter, tol):
    n_samples = X.shape[0]
    d = pairwise_l2_distance(X, X, square=True)
    d.flat[::n_samples + 1] = np.inf
    p = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        _, _, p[i] = binary_search_gamma(d[i], desired_perplexity, max_iter, tol)
    return p


def joint_probability(X, desired_perplexity, max_iter, tol):
    cond_p = conditional_probability(X, desired_perplexity, max_iter, tol)
    c = cond_p + cond_p.T
    sum_c = np.maximum(EPS, np.sum(c))
    p = np.maximum(EPS, c / sum_c)
    return p


def loss_gradient(p, Y):
    n_samples, n_components = Y.shape

    # q
    d = pairwise_l2_distance(Y, Y, square=True)
    c = 1 / (1 + d)
    c.flat[::n_samples + 1] = 0
    sum_c = np.maximum(EPS, np.sum(c))
    q = np.maximum(EPS, c / sum_c)

    # loss
    loss = np.sum(p * (np.log(p) - np.log(q)))

    # gradient
    p_q = p - q
    grad = np.zeros_like(Y)
    for k in range(n_components):
        grad[:, k] = 4 * np.sum(p_q * c * (Y[:, k, np.newaxis] - Y[:, k]), axis=1)

    return loss, {'Y': grad}


class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, max_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.optimizer = None
        self.embedding_ = None
        self.kl_divergence_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        p = joint_probability(X, self.perplexity, 100, 1e-4)

        self.embedding_ = 1e-4 * np.random.randn(n_samples, self.n_components)

        self.optimizer = MomentumGD(learning_rate=self.learning_rate)
        self.optimizer.minimize({'Y': self.embedding_},
                                lambda: loss_gradient(p, self.embedding_))

        self.optimizer.max_iter = 100
        self.optimizer.momentum = 0.5
        self.optimizer.run()

        self.optimizer.max_iter = self.max_iter
        self.optimizer.momentum = 0.8
        self.kl_divergence_ = self.optimizer.run()[-1]

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_
