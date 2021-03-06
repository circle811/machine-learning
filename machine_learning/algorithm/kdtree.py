import heapq
import numpy as np

from ..utils.distance import distance_function

__all__ = ['KDTree']


class Heap:
    def __init__(self):
        self._h = []

    def __len__(self):
        return len(self._h)

    def max(self):
        p = self._h[0]
        return p[1], -p[0]

    def items(self):
        l = [(data, -nv) for nv, data in self._h]
        l.sort(key=lambda it: (it[1], it[0]))
        return l

    def push(self, data, value):
        heapq.heappush(self._h, (-value, data))

    def pop_max(self):
        p = heapq.heappop(self._h)
        return p[1], -p[0]


class Inner:
    __slots__ = ['lower', 'upper', 'split_d', 'split_v', 'left', 'right']

    def __init__(self, lower, upper, split_d, split_v, left, right):
        self.lower = lower
        self.upper = upper
        self.split_d = split_d
        self.split_v = split_v
        self.left = left
        self.right = right


class Leaf:
    __slots__ = ['lower', 'upper', 'points']

    def __init__(self, lower, upper, points):
        self.lower = lower
        self.upper = upper
        self.points = points


class KDTree:
    def __init__(self, X, leaf_size=20):
        """
        :param X: array of float (n_samples * n_features)
            Data points of the kd tree.

        :param leaf_size: int (default=20)
            Leaf size of the kd tree.
        """

        n, kd = X.shape
        self.X = X
        self.leaf_size = leaf_size
        self.root = self._build_tree(np.full(kd, -np.inf),
                                     np.full(kd, np.inf),
                                     np.arange(n),
                                     0)

    def _build_tree(self, lower, upper, points, split_d):
        if points.shape[0] <= self.leaf_size:
            return Leaf(lower, upper, points)

        r = self._find_split(points, split_d)
        if r is None:
            return Leaf(lower, upper, points)

        c_split_d, split_v, mask = r

        l_upper = upper.copy()
        l_upper[c_split_d] = split_v
        r_lower = lower.copy()
        r_lower[c_split_d] = split_v

        next_split_d = (c_split_d + 1) % self.X.shape[1]

        return Inner(lower, upper, c_split_d, split_v,
                     self._build_tree(lower, l_upper, points[mask], next_split_d),
                     self._build_tree(r_lower, upper, points[~mask], next_split_d))

    def _find_split(self, points, split_d):
        kd = self.X.shape[1]
        for i in range(kd):
            c_split_d = (split_d + i) % kd
            s = self.X[points, c_split_d]
            split_v = np.median(s)
            mask = s < split_v
            if not np.all(mask) and np.any(mask):
                return c_split_d, split_v, mask

    def query(self, x, k, metric='l2_square'):
        """
        Find k nearest neighbors.

        :param x: array of float (n_features)
            Center point.

        :param k: int
            Number of neighbors.

        :param metric: string (default="l2_square")
            Distance metric, "l1", "l2", "l2_square" or "linf".

        :return: array of int (k), array of float (k)
            Indexes and distances of the points.
        """

        heap = Heap()
        self._search_k(x, k, distance_function[metric], heap, self.root)
        l = heap.items()
        return np.array([it[0] for it in l]), np.array([it[1] for it in l])

    def _search_k(self, x, k, distance, heap, node):
        if len(heap) == k and distance(x, np.clip(x, node.lower, node.upper)) >= heap.max()[1]:
            return

        if isinstance(node, Inner):
            if x[node.split_d] < node.split_v:
                self._search_k(x, k, distance, heap, node.left)
                self._search_k(x, k, distance, heap, node.right)
            else:
                self._search_k(x, k, distance, heap, node.right)
                self._search_k(x, k, distance, heap, node.left)
        else:
            d = distance(x, self.X[node.points])
            for i in range(node.points.shape[0]):
                if len(heap) < k:
                    heap.push(node.points[i], d[i])
                elif d[i] < heap.max()[1]:
                    heap.pop_max()
                    heap.push(node.points[i], d[i])

    def query_radius(self, x, r, metric='l2_square'):
        """
        Find neighbors where the distance between them to x less than of equal to r.

        :param x: array of float (n_features)
            Center point.

        :param r: float
            Radius.

        :param metric: string (default="l2_square")
            Distance metric, "l1", "l2", "l2_square" or "linf".

        :return: array of int (k), array of float (k)
            Indexes and distances of the points. Where k is the number of neighbors.
        """

        l = []
        self._search_r(x, r, distance_function[metric], l, self.root)
        i = np.concatenate([it[0] for it in l])
        d = np.concatenate([it[1] for it in l])
        si = np.argsort(d)
        return i[si], d[si]

    def _search_r(self, x, r, distance, l, node):
        if distance(x, np.clip(x, node.lower, node.upper)) > r:
            return

        if isinstance(node, Inner):
            self._search_r(x, r, distance, l, node.left)
            self._search_r(x, r, distance, l, node.right)
        else:
            d = distance(x, self.X[node.points])
            mask = d <= r
            l.append([node.points[mask], d[mask]])
