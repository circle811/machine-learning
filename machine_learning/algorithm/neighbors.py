import numpy as np

from .kdtree import KDTree
from ..utils.distance import pairwise_distance_function


class NearestNeighbors:
    def __init__(self, n_neighbors=5, radius=1.0, metric='l2_square', algorithm='kd_tree', leaf_size=20):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self._fit_X = None
        self._tree = None

    def fit(self, X):
        self._fit_X = X
        if self.algorithm == 'kd_tree':
            self._tree = KDTree(X, self.leaf_size)
        elif self.algorithm != 'brute':
            raise ValueError('algorithm')

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        n_samples = X.shape[0]
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if self.algorithm == 'kd_tree':
            i_neighbor = np.zeros((n_samples, n_neighbors), dtype=np.int64)
            if return_distance:
                d_neighbor = np.zeros((n_samples, n_neighbors))
                for i in range(n_samples):
                    i_neighbor[i], d_neighbor[i] = self._tree.query(X[i], n_neighbors, self.metric)
                return d_neighbor, i_neighbor
            else:
                for i in range(n_samples):
                    i_neighbor[i], _ = self._tree.query(X[i], n_neighbors, self.metric)
                return i_neighbor
        else:
            d = pairwise_distance_function[self.metric](X, self._fit_X)
            i_neighbor = np.argsort(d, axis=1)[:, :n_neighbors]
            if return_distance:
                d_neighbor = d[np.arange(n_samples)[:, np.newaxis], i_neighbor]
                return d_neighbor, i_neighbor
            else:
                return i_neighbor

    def radius_neighbors(self, X, radius=None, return_distance=True):
        n_samples = X.shape[0]
        if radius is None:
            radius = self.radius
        if self.algorithm == 'kd_tree':
            i_neighbor = np.zeros(n_samples, dtype=np.object_)
            if return_distance:
                d_neighbor = np.zeros(n_samples, dtype=np.object_)
                for i in range(n_samples):
                    i_neighbor[i], d_neighbor[i] = self._tree.query_radius(X[i], radius, self.metric)
                return d_neighbor, i_neighbor
            else:
                for i in range(n_samples):
                    i_neighbor[i], _ = self._tree.query(X[i], radius, self.metric)
                return i_neighbor
        else:
            d = pairwise_distance_function[self.metric](X, self._fit_X)
            i_neighbor = np.zeros(n_samples, dtype=np.object_)
            if return_distance:
                d_neighbor = np.zeros(n_samples, dtype=np.object_)
                for i in range(n_samples):
                    i_neighbor[i] = np.where(d[i] <= radius)[0]
                    d_neighbor[i] = d[i, i_neighbor[i]]
                return d_neighbor, i_neighbor
            else:
                for i in range(n_samples):
                    i_neighbor[i] = np.where(d[i] <= radius)[0]
                return i_neighbor
