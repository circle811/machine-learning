import numpy as np

from ..utils.kernel import kernel_function

__all__ = ['PCA', 'KernelPCA']

TINY = np.finfo(np.float64).tiny


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.n_components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.components_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)
        else:
            self.n_components_ = min(n_samples, n_features, self.n_components)
        self.mean_ = np.mean(X, axis=0)
        Xm = X - self.mean_
        u, s, v = np.linalg.svd(Xm, full_matrices=False)
        var = np.square(s) / (n_samples - 1)
        self.explained_variance_ = var[:self.n_components_]
        self.explained_variance_ratio_ = var[:self.n_components_] / np.maximum(TINY, np.sum(var))
        self.components_ = v[:self.n_components_]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        return X @ self.components_ + self.mean_


class KernelPCA:
    def __init__(self, n_components=None, kernel='rbf', degree=3, gamma=1.0, coef0=1.0,
                 fit_inverse_transform=False, alpha=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.fit_inverse_transform = fit_inverse_transform
        self.alpha = alpha
        self.n_components_ = None
        self.X_fit_ = None
        self.K_fit_rows_ = None
        self.K_fit_all_ = None
        self.lambdas_ = None
        self.alphas_ = None
        self.X_transformed_fit_ = None
        self.dual_coef_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.X_fit_ = X
        K = self._kernel_func(X, X)

        # center
        self.K_fit_rows_ = np.mean(K, axis=0)
        self.K_fit_all_ = np.mean(K)
        Km = K - self.K_fit_rows_[:, np.newaxis] - self.K_fit_rows_ + self.K_fit_all_

        # fit
        w, v = np.linalg.eigh(Km)
        pos = np.where(w > 0)[0]
        w = w[pos]
        v = v[:, pos]
        if self.n_components is None:
            self.n_components_ = pos.shape[0]
        else:
            self.n_components_ = min(pos.shape[0], self.n_components)
        self.lambdas_ = w[-1:-1 - self.n_components_:-1]
        self.alphas_ = v[:, -1:-1 - self.n_components_:-1]
        self.X_transformed_fit_ = self.alphas_ * np.sqrt(self.lambdas_)

        # fit inverse
        if self.fit_inverse_transform:
            Kt = self._kernel_func(self.X_transformed_fit_, self.X_transformed_fit_)
            self.dual_coef_ = np.linalg.pinv(Kt + self.alpha * np.eye(n_samples)) @ X

    def fit_transform(self, X):
        self.fit(X)
        return self.X_transformed_fit_

    def transform(self, X):
        K = self._kernel_func(X, self.X_fit_)
        Km = K - np.mean(K, axis=1)[:, np.newaxis] - self.K_fit_rows_ + self.K_fit_all_
        return Km @ (self.alphas_ / np.sqrt(self.lambdas_))

    def inverse_transform(self, X):
        if self.fit_inverse_transform:
            K = self._kernel_func(X, self.X_transformed_fit_)
            return K @ self.dual_coef_

    def _kernel_func(self, X, Y):
        if self.kernel == 'precomputed':
            return X
        params_dict = {
            'linear': {},
            'polynomial': {'degree': self.degree, 'gamma': self.gamma, 'coef0': self.coef0},
            'sigmoid': {'gamma': self.gamma, 'coef0': self.coef0},
            'rbf': {'gamma': self.gamma}
        }
        return kernel_function[self.kernel](X, Y, **params_dict[self.kernel])
