import numpy as np

__all__ = ['floyd_warshall']


def floyd_warshall(dist_matrix):
    """
    Find length of the shortest path between vertexes.

    :param dist_matrix: array of float (n_samples * n_samples)
        Adjacency matrix.

    :return: array of float (n_samples * n_samples)
        Length of the shortest path between vertexes.
    """

    d = dist_matrix
    for i in range(dist_matrix.shape[0]):
        d = np.minimum(d, d[:, i, np.newaxis] + d[i])
    return d
