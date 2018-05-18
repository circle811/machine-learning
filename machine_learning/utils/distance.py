import numpy as np

__all__ = ['l1_distance', 'l2_distance', 'linf_distance',
           'pairwise_l1_distance', 'pairwise_l2_distance', 'pairwise_linf_distance',
           'distance_function', 'pairwise_distance_function']


def l1_distance(x, y):
    """
    L1 distance.

    :param x: array of float (... * n_features)
        Points.

    :param y: array of float (... * n_features)
        Points.

    :return: array of float (...)
        L1 distance related to x and y.
    """

    return np.sum(np.abs(x - y), axis=-1)


def l2_distance(x, y, square=False):
    """
    L2 distance.

    :param x: array of float (... * n_features)
        Points.

    :param y: array of float (... * n_features)
        Points.

    :param square: bool (default=False)
        Whether return squared distance.

    :return: array of float (...)
        L2 distance (squared L2 distance) related to x and y.
    """

    if square:
        return np.sum(np.square(x - y), axis=-1)
    else:
        return np.sqrt(np.sum(np.square(x - y), axis=-1))


def linf_distance(x, y):
    """
    L infinity distance.

    :param x: array of float (... * n_features)
        Points.

    :param y: array of float (... * n_features)
        Points.

    :return: array of float (...)
        L infinity distance related to x and y.
    """

    return np.max(np.abs(x - y), axis=-1)


def pairwise_l1_distance(X, Y):
    """
    Pairwise L1 distance.

    :param X: array of float (m * n_features)
        Points.

    :param Y: array of float (n * n_features)
        Points.

    :return: array of float (m * n)
        L1 distance matrix.
    """

    m = X.shape[0]
    n = Y.shape[0]
    d = np.zeros((m, n))
    for i in range(n):
        d[:, i] = l1_distance(X, Y[i])
    return d


def pairwise_l2_distance(X, Y, square=False):
    """
    Pairwise L2 distance.

    :param X: array of float (m * n_features)
        Points.

    :param Y: array of float (n * n_features)
        Points.

    :param square: bool (default=False)
        Whether return squared distance.

    :return: array of float (m * n)
        L2 distance (squared L2 distance) matrix.
    """

    d2 = np.sum(np.square(X), axis=1)[:, np.newaxis] + np.sum(np.square(Y), axis=1) - 2 * (X @ Y.T)
    d2 = np.maximum(0, d2)
    if square:
        return d2
    else:
        return np.sqrt(d2)


def pairwise_linf_distance(X, Y):
    """
    Pairwise L infinity distance.

    :param X: array of float (m * n_features)
        Points.

    :param Y: array of float (n * n_features)
        Points.

    :return: array of float (m * n)
        L infinity distance matrix.
    """

    m = X.shape[0]
    n = Y.shape[0]
    d = np.zeros((m, n))
    for i in range(n):
        d[:, i] = linf_distance(X, Y[i])
    return d


distance_function = {
    'l1': l1_distance,
    'l2': l2_distance,
    'l2_square': lambda x, y: l2_distance(x, y, square=True),
    'linf': linf_distance
}

pairwise_distance_function = {
    'l1': pairwise_l1_distance,
    'l2': pairwise_l2_distance,
    'l2_square': lambda X, Y: pairwise_l2_distance(X, Y, square=True),
    'linf': pairwise_linf_distance
}
