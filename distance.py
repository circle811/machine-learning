import numpy as np

__all__ = ['l1_distance', 'l2_distance', 'linf_distance',
           'pairwise_l1_distance', 'pairwise_l2_distance', 'pairwise_linf_distance',
           'distance_function', 'pairwise_distance_function']


def l1_distance(x, y):
    return np.sum(np.abs(x - y), axis=-1)


def l2_distance(x, y, square=False):
    if square:
        return np.sum(np.square(x - y), axis=-1)
    else:
        return np.sqrt(np.sum(np.square(x - y), axis=-1))


def linf_distance(x, y):
    return np.max(np.abs(x - y), axis=-1)


def pairwise_l1_distance(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    d = np.zeros((m, n))
    for i in range(n):
        d[:, i] = l1_distance(X, Y[i])
    return d


def pairwise_l2_distance(X, Y, square=False):
    m = X.shape[0]
    n = Y.shape[0]
    d = np.zeros((m, n))
    for i in range(n):
        d[:, i] = l2_distance(X, Y[i], square)
    return d


def pairwise_linf_distance(X, Y):
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
