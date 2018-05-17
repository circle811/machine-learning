import numpy as np

from ..algorithm.optimizer import SGD, Adam

__all__ = ['NeuralNetworkClassifier', 'NeuralNetworkRegressor']


def to_dict(ws, bs):
    d = {}
    for i in range(ws.shape[0]):
        d['w{}'.format(i)] = ws[i]
        d['b{}'.format(i)] = bs[i]
    return d


class NeuralNetworkBase:
    def __init__(self, hidden_layer_sizes=(100,), alpha=1e-4, optimizer=None):
        """
        :param hidden_layer_sizes: tuple of int (default=(100,))
            Hidden layer sizes of the neural network.

        :param alpha: float (default=1e-4)
            Regular loss parameter.

        :param optimizer: optimizer object (default=Adam())
            Optimizer to minimize the loss function.
        """

        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = Adam()
        self.coefs_ = None
        self.intercepts_ = None
        self.loss_curve_ = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        layer_size = (n_features,) + self.hidden_layer_sizes + (self._output_size(),)
        n_layers = len(layer_size) - 1
        self.coefs_ = np.zeros(n_layers, dtype=np.object_)
        self.intercepts_ = np.zeros(n_layers, dtype=np.object_)
        for i in range(n_layers):
            n_in, n_out = layer_size[i], layer_size[i + 1]
            self.coefs_[i] = np.sqrt(2 / n_in) * np.random.randn(n_in, n_out)
            self.intercepts_[i] = np.sqrt(2 / n_in) * np.random.randn(n_out)
        self.optimizer.minimize(to_dict(self.coefs_, self.intercepts_),
                                lambda a=slice(None): self._loss_gradient(X[a], Y[a]))
        self.loss_curve_ = self.optimizer.run(n_samples=n_samples)

    def predict(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        if isinstance(self.optimizer, SGD):
            batch_size = self.optimizer.batch_size
        else:
            batch_size = 200
        n_samples = X.shape[0]
        return np.vstack([self._forward(X[i: i + batch_size])
                          for i in range(0, n_samples, batch_size)])

    def _output_size(self):
        raise NotImplementedError

    def _forward(self, X):
        n_layers = self.coefs_.shape[0]
        s = X
        for i in range(n_layers - 1):
            s = np.maximum(0, s @ self.coefs_[i] + self.intercepts_[i])
        s = s @ self.coefs_[-1] + self.intercepts_[-1]
        return s

    def _loss_gradient(self, X, Y):
        # forward pass
        n_layers = self.coefs_.shape[0]
        s = X
        ss = [s]
        for i in range(n_layers - 1):
            s = np.maximum(0, s @ self.coefs_[i] + self.intercepts_[i])
            ss.append(s)
        s = s @ self.coefs_[-1] + self.intercepts_[-1]

        # backward pass
        loss, grad_s = self._loss_gradient_output(s, Y)
        reg_loss = (0.5 * self.alpha) * np.sum([np.sum(np.square(w)) for w in self.coefs_])

        grad_w = np.zeros(n_layers, dtype=np.object_)
        grad_b = np.zeros(n_layers, dtype=np.object_)
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
    def __init__(self, hidden_layer_sizes=(100,), alpha=1e-4, optimizer=None):
        """
        :param hidden_layer_sizes: tuple of int (default=(100,))
            Hidden layer sizes of the neural network.

        :param alpha: float (default=1e-4)
            Regular loss parameter.

        :param optimizer: optimizer object (default=Adam())
            Optimizer to minimize the loss function.
        """

        super().__init__(hidden_layer_sizes, alpha, optimizer)
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
        d = s - Y[:, np.newaxis]
        loss = 0.5 * np.mean(np.square(d))
        grad = (1 / n_samples) * d
        return loss, grad
