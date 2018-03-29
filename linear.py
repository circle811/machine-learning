import numpy as np

from optimizer import AdamOptimizer

__all__ = ['LogisticRegression', 'LinearRegression']


class LinearBase:
    def __init__(self, alpha=1.0, l1_ratio=0.0, optimizer=None, batch_size=200, max_iter=200):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamOptimizer()
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.coefs_ = None
        self.intercepts_ = None
        self.loss_curve_ = None

    def fit(self, X, Y):
        n_features = X.shape[1]
        n_output = self._output_size()
        self.coefs_ = np.zeros((n_features, n_output))
        self.intercepts_ = np.zeros(n_output)
        parameters = {'w': self.coefs_, 'b': self.intercepts_}
        self.optimizer.minimize(parameters, self._loss_gradient)
        self.loss_curve_ = self.optimizer.run(X, Y, self.batch_size, self.max_iter)

    def predict(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        return X @ self.coefs_ + self.intercepts_

    def _output_size(self):
        raise NotImplementedError

    def _loss_gradient(self, X, Y):
        s = self.decision_function(X)
        loss, grad_s = self._loss_gradient_output(s, Y)
        reg_loss = self.alpha * (self.l1_ratio * np.sum(np.abs(self.coefs_)) +
                                 (1 - self.l1_ratio) * 0.5 * np.sum(np.square(self.coefs_)))
        reg_grad_w = self.alpha * (self.l1_ratio + np.where(self.coefs_ > 0, 1.0, -1.0) +
                                   (1 - self.l1_ratio) * self.coefs_)
        grad = {
            'w': X.T @ grad_s + reg_grad_w,
            'b': np.sum(grad_s, axis=0)
        }
        return loss + reg_loss, grad

    def _loss_gradient_output(self, s, Y):
        raise NotImplementedError


class LogisticRegression(LinearBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        if self.alpha * self.l1_ratio == 0.0:
            n_samples, n_features = X.shape
            Xe = np.hstack([X, np.ones((n_samples, 1))])
            w = np.linalg.pinv(Xe.T @ Xe + self.alpha * np.eye(n_features + 1)) @ (Xe.T @ Y)
            self.coefs_ = w[:-1, np.newaxis]
            self.intercepts_ = w[-1, np.newaxis]
        else:
            super().fit(X, Y)

    def predict(self, X):
        return self.decision_function(X)[:, 0]

    def _output_size(self):
        return 1

    def _loss_gradient_output(self, s, Y):
        n_samples = s.shape[0]
        d = s[:, 0] - Y
        loss = 0.5 * np.mean(np.square(d))
        grad = ((1 / n_samples) * d)[:, np.newaxis]
        return loss, grad
