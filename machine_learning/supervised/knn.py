import numpy as np

from ..algorithm.neighbors import NearestNeighbors

__all__ = ['KNNClassifier', 'KNNRegressor']


def count2d(Y, n_classes):
    c = np.zeros((Y.shape[0], n_classes), dtype=np.int64)
    a = np.arange(Y.shape[0])
    for i in range(Y.shape[1]):
        c[a, Y[:, i]] += 1
    return c


class KNNBase(NearestNeighbors):
    def __init__(self, n_neighbors=5, metric='l2_square', algorithm='kd_tree', leaf_size=20):
        super().__init__(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm, leaf_size=leaf_size)
        self._Y = None

    def fit(self, X, Y):
        super().fit(X)
        self._Y = Y

    def predict(self, X):
        raise NotImplementedError


class KNNClassifier(KNNBase):
    def __init__(self, n_neighbors=5, metric='l2_square', algorithm='kd_tree', leaf_size=20):
        super().__init__(n_neighbors, metric, algorithm, leaf_size)
        self.classes_ = None

    def fit(self, X, Y):
        self.classes_, Yi = np.unique(Y, return_inverse=True)
        super().fit(X, Yi)

    def predict(self, X):
        c = self._count(X)
        return self.classes_[np.argmax(c, axis=1)]

    def predict_proba(self, X):
        c = self._count(X)
        return c / np.sum(c, axis=1)[:, np.newaxis]

    def _count(self, X):
        neighbors = self.kneighbors(X, return_distance=False)
        return count2d(self._Y[neighbors], self.classes_.shape[0])


class KNNRegressor(KNNBase):
    def predict(self, X):
        neighbors = self.kneighbors(X, return_distance=False)
        return np.mean(self._Y[neighbors], axis=1)
