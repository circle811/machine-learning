import numpy as np

from .lbfgs import minimize_lbfgs

__all__ = ['GD', 'MomentumGD', 'SGD', 'MomentumSGD', 'Adam', 'LBFGS']


class OptimizerBase:
    def __init__(self, **kwargs):
        """
        :param kwargs: dict
            Keyword arguments.
        """

        assert len(kwargs) == 0
        self.parameter = None
        self.function_gradient = None

    def minimize(self, parameter, function_gradient):
        """
        Set parameter and function_gradient.

        :param parameter: dict (string -> array of float)
             Parameter to optimize.

        :param function_gradient: function (dict (string -> array of float) -> float, dict (string -> array of float))
             Computer function value and gradient.
        """

        self.parameter = parameter
        self.function_gradient = function_gradient

    def run(self, **kwargs):
        """
        Optimization process.

        :param kwargs: dict
            Keyword arguments.

        :return: array of float (n_iters)
            Function values of searched points.
        """

        raise NotImplementedError


class GD(OptimizerBase):
    def __init__(self, learning_rate=1e-3, max_iter=200, tol=1e-4, **kwargs):
        """
        :param learning_rate: float (defalut=1e-3)
            Fixed step length.

        :param max_iter: int (default=200)
            Maximum number of iterations.

        :param tol: float (default=1e-4)
            Tolerance of function value.

        :param kwargs: dict
            Keyword arguments.
        """

        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def run(self, **kwargs):
        """
        Optimization process.

        :param kwargs: dict
            Keyword arguments.

        :return: array of float (n_iters)
            Function values of searched points.
        """

        fs = []
        for _ in range(self.max_iter):
            f, g = self.function_gradient()
            fs.append(f)
            self.update(g)
        return np.array(fs)

    def update(self, gradient):
        for k in self.parameter:
            self.parameter[k] -= self.learning_rate * gradient[k]


class SGD(GD):
    def __init__(self, learning_rate=1e-3, max_iter=200, tol=1e-4, batch_size=200, **kwargs):
        """
        :param learning_rate: float (defalut=1e-3)
            Fixed step length.

        :param max_iter: int (default=200)
            Maximum number of iterations.

        :param tol: float (default=1e-4)
            Tolerance of function value.

        :param batch_size: int (default=200)
            Size of random subset.

        :param kwargs: dict
            Keyword arguments.
        """

        super().__init__(learning_rate=learning_rate, max_iter=max_iter, tol=tol, **kwargs)
        self.batch_size = batch_size

    def run(self, n_samples, **kwargs):
        """
        Optimization process.

        :param n_samples: int
            Number of samples.

        :param kwargs: dict
            Keyword arguments.

        :return: array of float (n_iters)
            Function values of searched points.
        """

        batch_size = min(n_samples, self.batch_size)
        n_batchs = (n_samples + batch_size - 1) // batch_size
        fs = []
        for _ in range(self.max_iter):
            sum_f = 0.0
            for j in range(n_batchs):
                np.random.choice(n_samples, size=batch_size, replace=False)
                f, g = self.function_gradient()
                sum_f += f
                self.update(g)
            fs.append(sum_f / n_batchs)
        return np.array(fs)


class MomentumUpdate(GD):
    def __init__(self, momentum=0.9, nesterovs=True, **kwargs):
        """
        :param momentum: float (default=0.9)
            Momentum parameter.

        :param nesterovs: bool (default=True)
            Whether use nesterov's momentum.

        :param kwargs: dict
            Keyword arguments.
        """

        super().__init__(**kwargs)
        self.momentum = momentum
        self.nesterovs = nesterovs
        self.m = None

    def minimize(self, parameter, function_gradient):
        """
        Set parameter and function_gradient, initialize momentum.

        :param parameter: dict (string -> array of float)
             Parameter to optimize.

        :param function_gradient: function (dict (string -> array of float) -> float, dict (string -> array of float))
             Computer function value and gradient.
        """

        super().minimize(parameter, function_gradient)
        self.m = {k: np.zeros_like(v) for k, v in parameter.items()}

    def update(self, gradient):
        if self.nesterovs:
            for k in self.parameter:
                self.m[k] = self.momentum * (self.m[k] + gradient[k])
                self.parameter[k] -= self.learning_rate * (gradient[k] + self.m[k])
        else:
            for k in self.parameter:
                self.m[k] = self.momentum * self.m[k] + gradient[k]
                self.parameter[k] -= self.learning_rate * self.m[k]


class AdamUpdate(GD):
    def __init__(self, beta_1=0.9, beta_2=0.999, epsilon=1e-8, **kwargs):
        """
        :param beta_1: float (default=0.9)
             Momentum parameter.

        :param beta_2: float (default=0.999)
             Momentum squared parameter.

        :param epsilon: float (default=1e-8)
             To avoid dividing by zero.

        :param kwargs: dict
            Keyword arguments.
        """

        super().__init__(**kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = None
        self.m = None
        self.v = None

    def minimize(self, parameter, function_gradient):
        """
        Set parameter and function_gradient, initialize momentum.

        :param parameter: dict (string -> array of float)
             Parameter to optimize.

        :param function_gradient: function (dict (string -> array of float) -> float, dict (string -> array of float))
             Computer function value and gradient.
        """

        super().minimize(parameter, function_gradient)
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in parameter.items()}
        self.v = {k: np.zeros_like(v) for k, v in parameter.items()}

    def update(self, gradient):
        self.t += 1
        for k in self.parameter:
            self.m[k] = self.beta_1 * self.m[k] + (1 - self.beta_1) * gradient[k]
            self.v[k] = self.beta_2 * self.v[k] + (1 - self.beta_2) * np.square(gradient[k])
            mc = self.m[k] / (1 - self.beta_1 ** self.t)
            vc = self.v[k] / (1 - self.beta_2 ** self.t)
            self.parameter[k] -= self.learning_rate * (mc / (np.sqrt(vc) + self.epsilon))


class MomentumGD(MomentumUpdate, GD):
    def __init__(self, learning_rate=1e-3, max_iter=200, tol=1e-4,
                 momentum=0.9, nesterovs=True, **kwargs):
        """
        :param learning_rate: float (defalut=1e-3)
            Fixed step length.

        :param max_iter: int (default=200)
            Maximum number of iterations.

        :param tol: float (default=1e-4)
            Tolerance of function value.

        :param momentum: float (default=0.9)
            Momentum parameter.

        :param nesterovs: bool (default=True)
            Whether use nesterov's momentum.

        :param kwargs: dict
            Keyword arguments.
        """

        super().__init__(learning_rate=learning_rate, max_iter=max_iter, tol=tol,
                         momentum=momentum, nesterovs=nesterovs, **kwargs)


class MomentumSGD(MomentumUpdate, SGD):
    def __init__(self, learning_rate=1e-3, max_iter=200, tol=1e-4, batch_size=200,
                 momentum=0.9, nesterovs=True, **kwargs):
        """
        :param learning_rate: float (defalut=1e-3)
            Fixed step length.

        :param max_iter: int (default=200)
            Maximum number of iterations.

        :param tol: float (default=1e-4)
            Tolerance of function value.

        :param batch_size: int (default=200)
            Size of random subset.

        :param momentum: float (default=0.9)
            Momentum parameter.

        :param nesterovs: bool (default=True)
            Whether use nesterov's momentum.

        :param kwargs: dict
            Keyword arguments.
        """

        super().__init__(learning_rate=learning_rate, max_iter=max_iter, tol=tol, batch_size=batch_size,
                         momentum=momentum, nesterovs=nesterovs, **kwargs)


class Adam(AdamUpdate, SGD):
    def __init__(self, learning_rate=1e-3, max_iter=200, tol=1e-4, batch_size=200,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-8, **kwargs):
        """
        :param learning_rate: float (defalut=1e-3)
            Fixed step length.

        :param max_iter: int (default=200)
            Maximum number of iterations.

        :param tol: float (default=1e-4)
            Tolerance of function value.

        :param batch_size: int (default=200)
            Size of random subset.

        :param beta_1: float (default=0.9)
             Momentum parameter.

        :param beta_2: float (default=0.999)
             Momentum squared parameter.

        :param epsilon: float (default=1e-8)
             To avoid dividing by zero.

        :param kwargs: dict
            Keyword arguments.
        """

        super().__init__(learning_rate=learning_rate, max_iter=max_iter, tol=tol, batch_size=batch_size,
                         beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, **kwargs)


class LBFGS(OptimizerBase):
    def __init__(self, m=20, max_iter=200, bs_max_iter=20, tol=1e-4, g_tol=1e-4, **kwargs):
        """
        :param m: int (default=20)
            Number of points and their gradient saved.

        :param max_iter: int (default=200)
            Maximum number of iterations.

        :param bs_max_iter: int (default=20)
            Maximum number of iterations of the linear search.

        :param tol: float (default=1e-4)
            Tolerance of function value.

        :param g_tol: float (default=1e-4)
            Tolerance of gradient.

        :param kwargs: dict
            Keyword arguments.
        """

        super().__init__(**kwargs)
        self.m = m
        self.max_iter = max_iter
        self.bs_max_iter = bs_max_iter
        self.tol = tol
        self.g_tol = g_tol
        self.keys = None
        self.slices = None

    def minimize(self, parameter, function_gradient):
        """
        Set parameter and function_gradient, set keys and slices.

        :param parameter: dict (string -> array of float)
             Parameter to optimize.

        :param function_gradient: function (dict (string -> array of float) -> float, dict (string -> array of float))
             Computer function value and gradient.
        """

        super().minimize(parameter, function_gradient)
        self.keys = sorted(parameter.keys())
        pos = np.add.accumulate([parameter[k].size for k in self.keys])
        self.slices = [slice(pos[i - 1] if i > 0 else 0, pos[i]) for i in range(len(self.keys))]

    def run(self, **kwargs):
        """
        Optimization process.

        :param kwargs: dict
            Keyword arguments.

        :return: array of float (n_iters)
            Function values of searched points.
        """

        x, fs = minimize_lbfgs(self.function_gradient_wrap, self.dict_to_array(self.parameter),
                               self.m, self.max_iter, self.bs_max_iter, self.tol, self.g_tol)
        self.set_parameter(x)
        return fs

    def dict_to_array(self, d):
        return np.concatenate([d[k].flat for k in self.keys])

    def set_parameter(self, x):
        for i, k in enumerate(self.keys):
            s = x[self.slices[i]]
            d = self.parameter[k]
            np.copyto(d, s.reshape(d.shape))

    def function_gradient_wrap(self, x):
        self.set_parameter(x)
        f, g = self.function_gradient()
        return f, self.dict_to_array(g)
