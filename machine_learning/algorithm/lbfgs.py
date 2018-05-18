import collections
import numpy as np

__all__ = ['minimize_lbfgs']

EPS = np.finfo(np.float64).eps

C1 = 0.0001
C2 = 0.9


def binary_search_wolfe(function_gradient, x, f, d, g_d, max_iter):
    """
    Linear search, use binary search and wolfe condition.
    """

    t = 1.0
    lower = 0.0
    upper = np.inf
    for i in range(max_iter):
        dx = t * d
        x1 = x + dx
        f1, g1 = function_gradient(x1)
        if f1 > f + C1 * t * g_d:
            upper = t
            t = (lower + upper) / 2
        elif np.dot(g1, d) < C2 * g_d:
            lower = t
            if upper == np.inf:
                t = lower * 2
            else:
                t = (lower + upper) / 2
        else:
            break
    return x1, f1, g1, dx


def compute_d(dx_list, dg_list, rho_list, g):
    """
    Compute search direction.
    """

    m = len(dx_list)
    alpha = np.zeros(m)
    d = -g
    for i in range(m - 1, -1, -1):
        alpha[i] = rho_list[i] * np.dot(dx_list[i], d)
        d = d - alpha[i] * dg_list[i]
    for i in range(m):
        beta_i = rho_list[i] * np.dot(dg_list[i], d)
        d = d + (alpha[i] - beta_i) * dx_list[i]
    return d


def minimize_lbfgs(function_gradient, x0, m=20, max_iter=200, bs_max_iter=20, tol=1e-4, g_tol=1e-4):
    """
    L-BFGS Algorithm.

    :param function_gradient: function (array of float (d) -> float, array of float (d))
        Computer function value and gradient.

    :param x0: array of float (d)
        Initial point.

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

    :return: array of float (d), array of float (n_iters)
        Minimum point, and function values of searched points.
    """

    dx_list = collections.deque()
    dg_list = collections.deque()
    rho_list = collections.deque()

    x = x0
    f, g = function_gradient(x)
    d = -g
    fs = [f]

    for _ in range(max_iter):
        g_d = np.dot(g, d)
        if np.max(np.abs(g)) <= g_tol or -g_d <= g_tol:
            break

        x1, f1, g1, dx = binary_search_wolfe(function_gradient, x, f, d, g_d, bs_max_iter)
        dg = g1 - g

        dx_dg = np.dot(dx, dg)
        if (f - f1) / max(abs(f), abs(f1), 1) <= tol or abs(dx_dg) <= g_tol * EPS:
            break

        dx_list.append(dx)
        dg_list.append(dg)
        rho_list.append(1 / dx_dg)
        if len(dx_list) > m:
            dx_list.popleft()
            dg_list.popleft()
            rho_list.popleft()

        d1 = compute_d(dx_list, dg_list, rho_list, g1)
        x, f, g, d = x1, f1, g1, d1
        fs.append(f)

    return x, np.array(fs)
