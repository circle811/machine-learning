import numpy as np

__all__ = ['l1_distance', 'l2_distance', 'linf_distance']


def l1_distance(x, y):
    return np.sum(np.abs(x - y), axis=-1)


def l2_distance(x, y, square=False):
    if square:
        return np.sum(np.square(x - y), axis=-1)
    else:
        return np.sqrt(np.sum(np.square(x - y), axis=-1))


def linf_distance(x, y):
    return np.max(np.abs(x - y), axis=-1)
