import numpy as np

from ..utils.kernel import kernel_function

__all__ = ['SVMClassifier']


def optimize(K, Y, C, alpha, g, i):
    d1 = Y[i] * ((g[i] - Y[i]) - (g - Y))
    d2 = K[i, i] + K.diagonal() - 2 * K[i]

    delta_u = np.zeros_like(alpha)
    np.divide(-d1, d2, out=delta_u, where=d2 > 0)

    L = np.maximum(-alpha[i], np.where(Y == Y[i], alpha - C, -alpha))
    R = np.minimum(C - alpha[i], np.where(Y == Y[i], alpha, C - alpha))
    delta = np.clip(delta_u, L, R)
    delta[i] = 0

    descent = -(d1 * delta + 0.5 * d2 * delta * delta)
    j = np.argmax(descent)

    return j, delta[j], descent[j]


def update(K, Y, alpha, g, i, j, delta):
    alpha[i] += delta
    alpha[j] -= Y[i] * Y[j] * delta
    g += Y[i] * delta * (K[i] - K[j])


def compute_b(Y, C, alpha, g):
    js = np.where((alpha > 0) & (alpha < C))[0]
    if len(js) == 0:
        return 0.0
    else:
        return np.mean(Y[js] - g[js])


def smo(K, Y, C, tol):
    n_samples = K.shape[0]
    alpha = np.zeros(n_samples)
    g = np.zeros(n_samples)  # g == (alpha * Y) @ K

    while True:
        s = np.random.randint(n_samples)
        for t in range(n_samples):
            i = (s + t) % n_samples
            j, delta, descent = optimize(K, Y, C, alpha, g, i)
            if descent > tol:
                update(K, Y, alpha, g, i, j, delta)
                break
        else:
            return alpha, compute_b(Y, C, alpha, g)


class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=1.0, coef0=1.0, tol=1e-4, multi_class='ovr'):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.multi_class = multi_class
        self.dual_coef_ = None
        self.intercept_ = None
        self.support_vectors_ = None
        self.classes_ = None

    def fit(self, X, Y):
        self.classes_, Yi = np.unique(Y, return_inverse=True)
        n_classes = self.classes_.shape[0]
        K = self._kernel_func(X, X)
        if n_classes == 2:
            y = np.where(Yi == 0, 1, -1)
            alpha, b = smo(K, y, self.C, self.tol)
            dual_coef = (y * alpha)[:, np.newaxis]
            intercept = np.array([b])
        elif self.multi_class == 'ovr':
            dual_coef, intercept = self._fit_ovr(K, Yi)
        elif self.multi_class == 'ovo':
            dual_coef, intercept = self._fit_ovo(K, Yi)
        else:
            raise ValueError('multi_class')
        sv = np.where(np.any(dual_coef != 0, axis=1))
        self.dual_coef_ = dual_coef[sv]
        self.intercept_ = intercept
        self.support_vectors_ = X[sv]

    def predict(self, X):
        n_classes = self.classes_.shape[0]
        K = self._kernel_func(X, self.support_vectors_)
        scores = K @ self.dual_coef_ + self.intercept_
        if n_classes == 2:
            return self.classes_[np.where(scores[:, 0] > 0, 0, 1)]
        elif self.multi_class == 'ovr':
            return self._predict_ovr(scores)
        elif self.multi_class == 'ovo':
            return self._predict_ovo(scores)
        else:
            raise ValueError('multi_class')

    def _kernel_func(self, X, Y):
        params_dict = {
            'linear': {},
            'polynomial': {'degree': self.degree, 'gamma': self.gamma, 'coef0': self.coef0},
            'sigmoid': {'gamma': self.gamma, 'coef0': self.coef0},
            'rbf': {'gamma': self.gamma}
        }
        return kernel_function[self.kernel](X, Y, **params_dict[self.kernel])

    def _fit_ovr(self, K, Yi):
        n_samples = K.shape[0]
        n_classes = self.classes_.shape[0]
        dual_coef = np.zeros((n_samples, n_classes))
        intercept = np.zeros(n_classes)
        for i in range(n_classes):
            y = np.where(Yi == i, 1, -1)
            alpha, b = smo(K, y, self.C, self.tol)
            dual_coef[:, i] = y * alpha
            intercept[i] = b
        return dual_coef, intercept

    def _predict_ovr(self, scores):
        return self.classes_[np.argmax(scores, axis=1)]

    def _fit_ovo(self, K, Yi):
        n_samples = K.shape[0]
        n_classes = self.classes_.shape[0]
        n_pairs = n_classes * (n_classes - 1) // 2
        dual_coef = np.zeros((n_samples, n_pairs))
        intercept = np.zeros(n_pairs)
        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                pos = np.where(Yi == i)[0]
                neg = np.where(Yi == j)[0]
                indexes = np.concatenate([pos, neg])
                y = np.concatenate([np.ones_like(pos), -np.ones_like(neg)])
                alpha, neg = smo(K[indexes[:, np.newaxis], indexes], y, self.C, self.tol)
                dual_coef[indexes, k] = y * alpha
                intercept[k] = neg
                k += 1
        return dual_coef, intercept

    def _predict_ovo(self, scores):
        n_samples = scores.shape[0]
        n_classes = self.classes_.shape[0]
        a = np.arange(n_samples)

        vote = np.zeros((n_samples, n_classes))
        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                p = np.where(scores[:, k] > 0, i, j)
                vote[a, p] += 1
                k += 1

        max_vote = np.max(vote, axis=1)[:, np.newaxis]
        candidate = vote == max_vote

        vote_c = np.where(candidate, 0, -np.inf)
        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                s = candidate[:, i] * candidate[:, j] * scores[:, k]
                vote_c[:, i] += s
                vote_c[:, j] -= s
                k += 1

        return self.classes_[np.argmax(vote_c, axis=1)]
