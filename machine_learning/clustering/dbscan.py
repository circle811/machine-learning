import numpy as np

from ..algorithm.disjoint_set import DisjointSet
from ..algorithm.neighbors import NearestNeighbors

__all__ = ['DBSCAN']


class DBSCAN():
    def __init__(self, eps=0.5, min_samples=5, metric='l2', algorithm='kd_tree', leaf_size=20):
        """
        :param eps: float (default=0.5)
            Maximum distance between neighbors.

        :param min_samples: int (default=5)
            Minimum number of neighbors required as a core point.

        :param metric: string (default="l2")
            Distance metric, "l1", "l2", "l2_square" or "linf".

        :param algorithm: string (default="kd_tree")
            Algorithm, "kd_tree" or "brute".

        :param leaf_size: int (default=20)
            Leaf size of the kd tree. Used when algorithm="kd_tree".
        """

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.core_sample_indices_ = None
        self.components_ = None
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]

        nn = NearestNeighbors(n_neighbors=self.min_samples, radius=self.eps, metric=self.metric,
                              algorithm=self.algorithm, leaf_size=self.leaf_size)
        nn.fit(X)
        i_neighbor = nn.radius_neighbors(return_distance=False)

        core = np.array([i_neighbor[i].shape[0] >= self.min_samples for i in range(n_samples)])
        noisy = np.full(n_samples, False)

        s = DisjointSet(n_samples)
        for i in range(n_samples):
            if core[i]:
                for j in i_neighbor[i]:
                    if core[j]:
                        s.union(i, j)
            else:
                for j in i_neighbor[i]:
                    if core[j]:
                        s.union(i, j)
                        break
                else:
                    noisy[i] = True

        labels0 = np.where(noisy, n_samples, s.get_labels())
        _, labels1 = np.unique(labels0, return_inverse=True)

        self.core_sample_indices_ = np.where(core)[0]
        self.components_ = X[core]
        self.labels_ = np.where(noisy, -1, labels1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
