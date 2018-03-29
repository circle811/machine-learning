import numpy as np

from optimizer import AdamOptimizer

__all__ = ['NeuralNetworkClassifier', 'NeuralNetworkRegressor']


def to_dict(ws, bs):
    d = {}
    for i in range(len(ws)):
        d['w{}'.format(i)] = ws[i]
        d['b{}'.format(i)] = bs[i]
    return d


class NeuralNetworkBase:
    def __init__(self, hidden_layer_sizes=(100,), alpha=0.0001, optimizer=None, batch_size=200, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
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
        layer_size = X.shape[1:] + self.hidden_layer_sizes + (self._output_size(),)
        n_layers = len(layer_size) - 1
        self.coefs_ = []
        self.intercepts_ = []
        for i in range(n_layers):
            n_in, n_out = layer_size[i], layer_size[i + 1]
            self.coefs_.append(np.sqrt(2 / n_in) * np.random.randn(n_in, n_out))
            self.intercepts_.append(np.sqrt(2 / n_in) * np.random.randn(n_out))
        parameters = to_dict(self.coefs_, self.intercepts_)
        self.optimizer.minimize(parameters, self._loss_gradient)
        self.loss_curve_ = self.optimizer.run(X, Y, self.batch_size, self.max_iter)

    def predict(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        n_samples = X.shape[0]
        return np.vstack([self._forward(X[i: i + self.batch_size])
                          for i in range(0, n_samples, self.batch_size)])

    def _output_size(self):
        raise NotImplementedError

    def _forward(self, X):
        n_layers = len(self.coefs_)
        s = X
        for i in range(n_layers - 1):
            s = np.maximum(0, s @ self.coefs_[i] + self.intercepts_[i])
        s = s @ self.coefs_[-1] + self.intercepts_[-1]
        return s

    def _loss_gradient(self, X, Y):
        # forward pass
        n_layers = len(self.coefs_)
        s = X
        ss = [s]
        for i in range(n_layers - 1):
            s = np.maximum(0, s @ self.coefs_[i] + self.intercepts_[i])
            ss.append(s)
        s = s @ self.coefs_[-1] + self.intercepts_[-1]

        # backward pass
        loss, grad_s = self._loss_gradient_output(s, Y)
        reg_loss = (0.5 * self.alpha) * np.sum([np.sum(np.square(w)) for w in self.coefs_])

        grad_w = [None] * n_layers
        grad_b = [None] * n_layers
        for i in range(n_layers - 1, 0, -1):
            grad_w[i] = ss[i].T @ grad_s + self.alpha * self.coefs_[i]
            grad_b[i] = np.sum(grad_s, axis=0)
            grad_s = (grad_s @ self.coefs_[i].T) * (ss[i] > 0)
        grad_w[0] = ss[0].T @ grad_s + self.alpha * self.coefs_[0]
        grad_b[0] = np.sum(grad_s, axis=0)

        return loss + reg_loss, to_dict(grad_w, grad_b)

    def _loss_gradient_output(self, s, Y):
        raise NotImplementedError


class NeuralNetworkClassifier(NeuralNetworkBase):
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
        return self.classes_.shape[0]

    def _loss_gradient_output(self, s, Y):
        n_samples = s.shape[0]

        p = np.exp(s - np.max(s, axis=1)[:, np.newaxis])
        p /= np.sum(p, axis=1)[:, np.newaxis]

        a = np.arange(n_samples)
        loss = -np.mean(np.log(p[a, Y]))
        p[a, Y] -= 1
        grad = (1 / n_samples) * p
        return loss, grad


class NeuralNetworkRegressor(NeuralNetworkBase):
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
