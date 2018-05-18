import numpy as np

from ..algorithm.disjoint_set import DisjointSet
from ..utils.distance import pairwise_l2_distance, pairwise_distance_function

__all__ = ['AgglomerativeClustering']


class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='ward', affinity='l2'):
        """
        :param n_clusters: int (default=2)
            Number of clusters.

        :param linkage: string (default="ward")
            Linkage type, "ward", "single", "complete" or "average".

        :param affinity: string (default="l2")
            Distance metric, "l1", "l2", "l2_square" or "linf".
        """

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        a = np.arange(n_samples)

        distance = self._init_distance(X)
        distance[a, a] = np.inf
        count = np.ones(n_samples, dtype=np.int64)
        s = DisjointSet(n_samples)

        for _ in range(n_samples - self.n_clusters):
            i, j = divmod(np.argmin(distance), n_samples)
            d = self._distance_to_union(distance, count, i, j)
            distance[i] = d
            distance[:, i] = d
            distance[j] = np.inf
            distance[:, j] = np.inf
            count[i] += count[j]
            count[j] = 0
            s.union(i, j)

        self.labels_ = np.unique(s.get_labels(), return_inverse=True)[1]

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def _init_distance(self, X):
        if self.linkage == 'ward':
            distance = 0.5 * pairwise_l2_distance(X, X, square=True)
        else:
            distance = pairwise_distance_function[self.affinity](X, X)
        return distance

    def _distance_to_union(self, distance, count, i, j):
        if self.linkage == 'ward':
            d = (((count[i] + count) * distance[i]
                  + (count[j] + count) * distance[j]
                  - count * distance[i, j])
                 / (count[i] + count[j] + count))
        elif self.linkage == 'single':
            d = np.minimum(distance[i], distance[j])
        elif self.linkage == 'complete':
            d = np.maximum(distance[i], distance[j])
        elif self.linkage == 'average':
            d = (count[i] * distance[i] + count[j] * distance[j]) / (count[i] + count[j])
        else:
            raise ValueError('affinity')
        d[i] = np.inf
        d[j] = np.inf
        return d
