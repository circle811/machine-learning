import numpy as np

from distance import pairwise_distance_function
from kdtree import KDTree

__all__ = ['KNNClassifier', 'KNNRegressor']


def count2d(Y, n_classes):
    c = np.zeros((Y.shape[0], n_classes), dtype=np.int64)
    a = np.arange(Y.shape[0])
    for i in range(Y.shape[1]):
        c[a, Y[:, i]] += 1
    return c


class KNNBase:
    def __init__(self, n_neighbors=5, metric='l2_square', algorithm='kd_tree', leaf_size=20):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.X = None
        self.Y = None
        self.tree = None

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        if self.algorithm == 'kd_tree':
            self.tree = KDTree(X, self.leaf_size)
        elif self.algorithm != 'brute':
            raise ValueError('algorithm')

    def predict(self, X):
        raise NotImplementedError

    def _nearest_neighbors(self, X):
        if self.algorithm == 'kd_tree':
            return [self.tree.query(X[i], self.n_neighbors, self.metric)[0] for i in range(X.shape[0])]
        else:
            pairwise_distance = pairwise_distance_function[self.metric]
            d = pairwise_distance(X, self.X)
            return np.argsort(d, axis=1)[:, :self.n_neighbors]


class KNNClassifier(KNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        neighbors = self._nearest_neighbors(X)
        return count2d(self.Y[neighbors], self.classes_.shape[0])


class KNNRegressor(KNNBase):
    def predict(self, X):
        neighbors = self._nearest_neighbors(X)
        return np.mean(self.Y[neighbors], axis=1)
