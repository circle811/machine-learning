import numpy as np

from .distance import pairwise_l2_distance

__all__ = ['linear', 'polynomial', 'sigmoid', 'rbf', 'kernel_function']


def linear(X, Y):
    return X @ Y.T


def polynomial(X, Y, degree=3, gamma=1, coef0=1):
    return (gamma * (X @ Y.T) + coef0) ** degree


def sigmoid(X, Y, gamma=1, coef0=1):
    return np.tanh(gamma * (X @ Y.T) + coef0)


def rbf(X, Y, gamma=1):
    d2 = pairwise_l2_distance(X, Y, square=True)
    return np.exp(-gamma * d2)


kernel_function = {
    'linear': linear,
    'polynomial': polynomial,
    'sigmoid': sigmoid,
    'rbf': rbf
}
