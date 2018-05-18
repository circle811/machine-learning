import numpy as np

from .pca import KernelPCA
from ..algorithm.graph import floyd_warshall
from ..algorithm.neighbors import NearestNeighbors

__all__ = ['Isomap']


class Isomap:
    def __init__(self, n_components=2, n_neighbors=5, algorithm='kd_tree', leaf_size=20):
        """
        :param n_components: int or None (default=2)
            Number of Components.
            - if int,  n_components
            - if None, rank of kernel matrix

        :param n_neighbors: int (default=5)
            Number of neighbors.

        :param algorithm: string (default="kd_tree")
            Algorithm, "kd_tree" or "brute".

        :param leaf_size: int (default=20)
            Leaf size of the kd tree. Used when algorithm="kd_tree".
        """

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.nbrs_ = None
        self.kernel_pca_ = None
        self.training_data_ = None
        self.dist_matrix_ = None
        self.embedding_ = None

    def fit(self, X):
        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric='l2',
                                      algorithm=self.algorithm, leaf_size=self.leaf_size)
        self.kernel_pca_ = KernelPCA(n_components=self.n_components, kernel='precomputed')
        self.training_data_ = X
        self.nbrs_.fit(X)

        # distance matrix between neighbors
        dn = self.nbrs_.kneighbors_graph(mode='distance')
        dn[dn == 0] = np.inf
        dn.flat[::dn.shape[0] + 1] = 0
        dn = np.minimum(dn, dn.T)

        # distance matrix
        d = floyd_warshall(dn)
        d[d == np.inf] = 0
        self.dist_matrix_ = d

        # `k` is not the real kernel matrix, but it works after centering.
        k = -0.5 * np.square(self.dist_matrix_)
        self.embedding_ = self.kernel_pca_.fit_transform(k)

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_

    def transform(self, X):
        n_samples = X.shape[0]
        d_neighbor, i_neighbor = self.nbrs_.kneighbors(X, return_distance=True)

        # distance matrix from `X` to `self.training_data_`
        d = np.zeros((n_samples, self.training_data_.shape[0]))
        for i in range(n_samples):
            d[i] = np.min(d_neighbor[i, :, np.newaxis] + self.dist_matrix_[i_neighbor[i]], axis=0)

        # `k` is not the real kernel matrix, but it works after centering.
        k = -0.5 * np.square(d)
        return self.kernel_pca_.transform(k)
