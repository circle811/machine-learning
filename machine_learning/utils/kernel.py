import numpy as np

from .distance import pairwise_l2_distance

__all__ = ['linear', 'polynomial', 'sigmoid', 'rbf', 'kernel_function']


def linear(X, Y):
    """
    Linear kernel.

    :param X: array of float (m * n_features)
        Points.

    :param Y: array of float (n * n_features)
        Points.

    :return: array of float (m * n)
        Kernel matrix.
    """

    return X @ Y.T


def polynomial(X, Y, degree=3, gamma=1.0, coef0=1.0):
    """
    Polynomial kernel.

    :param X: array of float (m * n_features)
        Points.

    :param Y: array of float (n * n_features)
        Points.

    :param degree: int (default=3)
        Polynomial degree.

    :param gamma: float (default=1.0)
        Kernel coefficient.

    :param coef0: float (default=1.0)
        Kernel intercept.

    :return: array of float (m * n)
        Kernel matrix.
    """

    return (gamma * (X @ Y.T) + coef0) ** degree


def sigmoid(X, Y, gamma=1.0, coef0=1.0):
    """
    Sigmoid kernel.

    :param X: array of float (m * n_features)
        Points.

    :param Y: array of float (n * n_features)
        Points.

    :param gamma: float (default=1.0)
        Kernel coefficient.

    :param coef0: float (default=1.0)
        Kernel intercept.

    :return: array of float (m * n)
        Kernel matrix.
    """

    return np.tanh(gamma * (X @ Y.T) + coef0)


def rbf(X, Y, gamma=1.0):
    """
    RBF kernel.

    :param X: array of float (m * n_features)
        Points.

    :param Y: array of float (n * n_features)
        Points.

    :param gamma: float (default=1.0)
        Kernel coefficient.

    :return: array of float (m * n)
        Kernel matrix.
    """

    d2 = pairwise_l2_distance(X, Y, square=True)
    return np.exp(-gamma * d2)


kernel_function = {
    'linear': linear,
    'polynomial': polynomial,
    'sigmoid': sigmoid,
    'rbf': rbf
}
