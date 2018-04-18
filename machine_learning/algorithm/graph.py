import numpy as np

__all__ = ['floyd_warshall']


def floyd_warshall(dist_matrix):
    d = dist_matrix
    for i in range(dist_matrix.shape[0]):
        d = np.minimum(d, d[:, i, np.newaxis] + d[i])
    return d
