import numpy as np


def optimize_two(K, Y, C, alpha, g, i, j):
    # target = const + d1 * delta + 0.5 * d2 * delta ** 2
    # interval of delta alpha[i]
    if Y[i] == Y[j]:
        L = max(-alpha[i], alpha[j] - C)
        R = min(C - alpha[i], alpha[j])
    else:
        L = max(-alpha[i], -alpha[j])
        R = min(C - alpha[i], C - alpha[j])

    # first and second derivative of alpha[i]
    d1 = Y[i] * (g[i] - g[j] - Y[i] + Y[j])
    d2 = K[i, i] + K[j, j] - 2 * K[i, j]

    if d2 < 1e-10:
        return 0.0, 0.0

    delta = np.clip(-d1 / d2, L, R)
    return delta, -(d1 * delta + 0.5 * d2 * delta * delta)


def update_two(K, Y, alpha, g, i, j, delta):
    alpha[i] += delta
    alpha[j] += -Y[i] * Y[j] * delta
    g += delta * Y[i] * (K[i, :] - K[j, :])


def smo(K, Y, C, tol):
    n_samples = K.shape[0]
    alpha = np.zeros(n_samples)

    # g == np.sum((alpha * Y) * K, axis=1)
    g = np.zeros(n_samples)

    count = 0
    while True:
        for i in range(n_samples):
            best_j = None
            best_delta = None
            best_des = -1
            for j in range(n_samples):
                if j != i:
                    delta, des = optimize_two(K, Y, C, alpha, g, i, j)
                    if best_des < des:
                        best_j = j
                        best_delta = delta
                        best_des = des
            if best_des > tol:
                # print('update i={} j={} delta={} des={}'.format(i, best_j, best_delta, best_des))
                update_two(K, Y, alpha, g, i, best_j, best_delta)
                count = 0
            else:
                count += 1
                if count >= n_samples:
                    return alpha


def rbf_kernel(X, Z, gamma):
    if Z is None:
        Z = X
    x2 = np.sum(np.square(X), axis=1)
    z2 = np.sum(np.square(Z), axis=1)
    d2 = x2[:, None] + z2[None, :] - 2 * X @ Z.T
    return np.exp(-gamma * d2)


class SVM:
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0, tol=1e-5):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.dual_coef = None
        self.support_vectors = None
        self.intercept = None
        if kernel == 'rbf':
            self.kernel_func = lambda X, Z: rbf_kernel(X, Z, self.gamma)
        elif kernel == 'linear':
            self.kernel_func = lambda X, Z: X @ Z.T
        else:
            raise ValueError('unknown kernel')

    def fit(self, X, Y):
        K = self.kernel_func(X, X)
        alpha = smo(K, Y, self.C, self.tol)
        s = np.where(alpha > 0)
        self.dual_coef = (alpha * Y)[s]
        self.support_vectors = X[s]
        j = np.where((alpha > 0) & (alpha < self.C))[0]
        if len(j) == 0:
            self.intercept = 0.0
        else:
            k1 = self.kernel_func(X[j], self.support_vectors)
            bs = Y[j] - np.sum(self.dual_coef * k1, axis=1)
            self.intercept = np.median(bs)

    def predict(self, X):
        K = self.kernel_func(X, self.support_vectors)
        scores = np.sum(self.dual_coef * K, axis=1) + self.intercept
        return np.where(scores < 0, -1, 1)
