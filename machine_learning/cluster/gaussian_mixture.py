import numpy as np

from .kmeans import KMeans

__all__ = ['GaussianMixture']


def log_proba(weights, means, precisions, X):
    n_components = weights.shape[0]
    n_samples, n_features = X.shape
    c = -0.5 * np.log(2 * np.pi) * n_features
    lp = np.zeros((n_samples, n_components))
    for i in range(n_components):
        Xm = X - means[i]
        q = np.sum(Xm @ precisions[i] * Xm, axis=1)
        lp[:, i] = (np.log(weights[i]) +
                    c +
                    0.5 * np.log(np.linalg.det(precisions[i])) +
                    -0.5 * q)
    return lp


class GaussianMixture:
    def __init__(self, n_components=1, n_init=1, max_iter=100, tol=1e-4, reg_covar=1e-6):
        self.n_components = n_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.precisions_ = None
        self.lower_bound_ = None

    def fit(self, X):
        weights = None
        means = None
        covariances = None
        precisions = None
        lower_bound = -np.inf

        for _ in range(self.n_init):
            w, m, c, p, l = self._iter(X)
            if lower_bound < l:
                weights = w
                means = m
                covariances = c
                precisions = p
                lower_bound = l

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.precisions_ = precisions
        self.lower_bound_ = lower_bound

    def predict(self, X):
        lp = log_proba(self.weights_, self.means_, self.precisions_, X)
        return np.argmax(lp, axis=1)

    def predict_proba(self, X):
        lp = log_proba(self.weights_, self.means_, self.precisions_, X)
        mlp = np.max(lp, axis=1)[:, np.newaxis]
        p = np.exp(lp - mlp)
        sp = np.sum(p, axis=1)[:, np.newaxis]
        p /= sp
        return p

    def _iter(self, X):
        n_samples, n_features = X.shape
        eye = np.eye(n_features)

        weights = np.full(self.n_components, 1 / self.n_components)
        means = self._init_centers(X)
        covariances = None
        precisions = np.stack([eye] * self.n_components)
        lower_bound = -np.inf

        for _ in range(self.max_iter):
            # E step
            lp = log_proba(weights, means, precisions, X)
            mlp = np.max(lp, axis=1)[:, np.newaxis]
            p = np.exp(lp - mlp)
            sp = np.sum(p, axis=1)[:, np.newaxis]
            p /= sp

            new_lower_bound = np.mean(np.log(sp) + mlp)
            if new_lower_bound - lower_bound <= self.tol:
                break

            # M step
            new_weights = p.mean(axis=0)
            new_means = np.zeros((self.n_components, n_features))
            new_covariances = np.zeros((self.n_components, n_features, n_features))
            for i in range(self.n_components):
                w = p[:, i, np.newaxis]
                sw = np.sum(w)
                new_means[i] = np.sum(w * X, axis=0) / sw
                Xm = X - new_means[i]
                new_covariances[i] = (w * Xm).T @ Xm / sw + self.reg_covar * eye

            weights = new_weights
            means = new_means
            covariances = new_covariances
            precisions = np.linalg.inv(new_covariances)
            lower_bound = new_lower_bound

        return weights, means, covariances, precisions, lower_bound

    def _init_centers(self, X):
        km = KMeans(n_clusters=self.n_components)
        km.fit(X)
        return km.cluster_centers_
