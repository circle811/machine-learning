import numpy as np

from kdtree import KDTree


def count(Y, n_classes):
    c = np.zeros(n_classes, dtype=np.int64)
    for i in range(Y.shape[0]):
        c[Y[i]] += 1
    return c


class KNNBase:
    def __init__(self, n_neighbors=5, metric='l2_square', leaf_size=20):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.leaf_size = leaf_size
        self.tree = None
        self.Y = None

    def fit(self, X, Y):
        self.tree = KDTree(X, self.leaf_size)
        self.Y = Y

    def predict(self, X):
        Y = []
        for i in range(X.shape[0]):
            neighbors, _ = self.tree.query(X[i], self.n_neighbors, self.metric)
            Y.append(self._accumulate(neighbors))
        return np.stack(Y)

    def _accumulate(self, neighbors):
        raise NotImplementedError


class KNNClassifier(KNNBase):
    def __init__(self, n_neighbors=5, metric='l2_square', leaf_size=20):
        super().__init__(n_neighbors, metric, leaf_size)
        self.classes_ = None

    def fit(self, X, Y):
        self.classes_, Yi = np.unique(Y, return_inverse=True)
        super().fit(X, Yi)

    def predict(self, X):
        c = super().predict(X)
        return self.classes_[np.argmax(c, axis=1)]

    def predict_proba(self, X):
        c = super().predict(X)
        return c / np.sum(c, axis=1)[:, np.newaxis]

    def _accumulate(self, neighbors):
        return count(self.Y[neighbors], self.classes_.shape[0])


class KNNRegressor(KNNBase):
    def _accumulate(self, neighbors):
        return np.mean(self.Y[neighbors])
