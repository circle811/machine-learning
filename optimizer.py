import numpy as np

__all__ = ['SGDOptimizer', 'AdamOptimizer']


class SGDBase:
    def __init__(self):
        self.parameters = None
        self.loss_gradient = None

    def minimize(self, parameters, loss_gradient):
        self.parameters = parameters
        self.loss_gradient = loss_gradient

    def run(self, X, Y, batch_size=200, max_iter=200):
        n_samples = X.shape[0]
        a = np.arange(n_samples)
        loss_curve = []
        best_loss = np.inf
        count = 0
        for it in range(max_iter):
            np.random.shuffle(a)
            total_loss = 0
            for i in range(0, n_samples, batch_size):
                b = a[i:i + batch_size]
                loss, grad = self.loss_gradient(X[b], Y[b])
                self.apply_gradient(grad)
                total_loss += loss * b.shape[0]
            total_loss /= n_samples
            loss_curve.append(total_loss)
            if best_loss > total_loss:
                best_loss = total_loss
                count = 0
            else:
                count += 1
            if count > 4:
                break

        return loss_curve

    def apply_gradient(self, gradients):
        raise NotImplementedError


class SGDOptimizer(SGDBase):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate

    def apply_gradient(self, gradients):
        for k in self.parameters:
            self.parameters[k] -= self.learning_rate * gradients[k]


class AdamOptimizer(SGDBase):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = None
        self.ms = None
        self.vs = None

    def minimize(self, parameters, loss_gradient):
        super().minimize(parameters, loss_gradient)
        self.t = 0
        self.ms = {k: np.zeros_like(v) for k, v in parameters.items()}
        self.vs = {k: np.zeros_like(v) for k, v in parameters.items()}

    def apply_gradient(self, gradients):
        self.t += 1
        for k in self.parameters:
            self.update(self.parameters[k], gradients[k], self.ms[k], self.vs[k])

    def update(self, p, g, m, v):
        m = self.beta_1 * m + (1 - self.beta_1) * g
        v = self.beta_2 * v + (1 - self.beta_2) * np.square(g)
        mc = m / (1 - self.beta_1 ** self.t)
        vc = v / (1 - self.beta_2 ** self.t)
        p -= self.learning_rate * (mc / (np.sqrt(vc) + self.epsilon))
