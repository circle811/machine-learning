import numpy as np


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.n_components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.components_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        if self.n_components is None:
            self.n_components_ = min(X.shape)
        else:
            self.n_components_ = self.n_components
        self.mean_ = np.mean(X, axis=0)
        Xm = X - self.mean_
        u, s, v = np.linalg.svd(Xm, full_matrices=False)
        self.explained_variance_ = np.square(s[:self.n_components_]) / (n_samples - 1)
        self.components_ = v[:self.n_components_]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        return X @ self.components_ + self.mean_
