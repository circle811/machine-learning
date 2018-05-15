import numpy as np

from ..algorithm.optimizer import LBFGS

__all__ = ['LogisticRegression', 'LinearRegression']


class LinearBase:
    def __init__(self, alpha=1e-4, l1_ratio=0.0, optimizer=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = LBFGS()
        self.coef_ = None
        self.intercept_ = None
        self.loss_curve_ = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        n_output = self._output_size()
        self.coef_ = np.zeros((n_features, n_output))
        self.intercept_ = np.zeros(n_output)
        self.optimizer.minimize({'w': self.coef_, 'b': self.intercept_},
                                lambda a=slice(None): self._loss_gradient(X[a], Y[a]))
        self.loss_curve_ = self.optimizer.run(n_samples=n_samples)

    def predict(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        return X @ self.coef_ + self.intercept_

    def _output_size(self):
        raise NotImplementedError

    def _loss_gradient(self, X, Y):
        s = X @ self.coef_ + self.intercept_
        loss, grad_s = self._loss_gradient_output(s, Y)
        reg_loss = self.alpha * (self.l1_ratio * np.sum(np.abs(self.coef_)) +
                                 (1 - self.l1_ratio) * 0.5 * np.sum(np.square(self.coef_)))
        reg_grad_w = self.alpha * (self.l1_ratio * np.where(self.coef_ > 0, 1.0, -1.0) +
                                   (1 - self.l1_ratio) * self.coef_)
        grad = {
            'w': X.T @ grad_s + reg_grad_w,
            'b': np.sum(grad_s, axis=0)
        }
        return loss + reg_loss, grad

    def _loss_gradient_output(self, s, Y):
        raise NotImplementedError


class LogisticRegression(LinearBase):
    def __init__(self, alpha=1e-4, l1_ratio=0.0, optimizer=None):
        super().__init__(alpha, l1_ratio, optimizer)
        self.classes_ = None

    def fit(self, X, Y):
        self.classes_, Yi = np.unique(Y, return_inverse=True)
        super().fit(X, Yi)

    def predict(self, X):
        s = self.decision_function(X)
        return self.classes_[np.argmax(s, axis=1)]

    def predict_proba(self, X):
        s = self.decision_function(X)
        p = np.exp(s - np.max(s, axis=1)[:, np.newaxis])
        p /= np.sum(p, axis=1)[:, np.newaxis]
        return p

    def decision_function(self, X):
        s = super().decision_function(X)
        return np.hstack([s, np.zeros((X.shape[0], 1))])

    def _output_size(self):
        return self.classes_.shape[0] - 1

    def _loss_gradient_output(self, s, Y):
        n_samples = s.shape[0]
        s = np.hstack([s, np.zeros((n_samples, 1))])

        p = np.exp(s - np.max(s, axis=1)[:, np.newaxis])
        p /= np.sum(p, axis=1)[:, np.newaxis]

        a = np.arange(n_samples)
        loss = -np.mean(np.log(p[a, Y]))
        p[a, Y] -= 1
        grad = (1 / n_samples) * p
        return loss, grad[:, :-1]


class LinearRegression(LinearBase):
    def fit(self, X, Y):
        if self.alpha * self.l1_ratio == 0:
            n_samples, n_features = X.shape
            Xe = np.hstack([X, np.ones((n_samples, 1))])
            w = np.linalg.pinv(Xe.T @ Xe + n_samples * self.alpha * np.eye(n_features + 1)) @ (Xe.T @ Y)
            self.coef_ = w[:-1, np.newaxis]
            self.intercept_ = w[-1, np.newaxis]
        else:
            super().fit(X, Y)

    def predict(self, X):
        return self.decision_function(X)[:, 0]

    def _output_size(self):
        return 1

    def _loss_gradient_output(self, s, Y):
        n_samples = s.shape[0]
        d = s - Y[:, np.newaxis]
        loss = 0.5 * np.mean(np.square(d))
        grad = (1 / n_samples) * d
        return loss, grad
