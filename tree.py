import numpy as np


def count(Y):
    n_classes = np.max(Y) + 1
    c = np.zeros(n_classes, dtype=np.int64)
    for i in range(len(Y)):
        c[Y[i]] += 1
    return c


def gini(c):
    return 1 - np.sum(np.square(c / np.sum(c)))


def entropy(c):
    c = np.array(c)
    p = c[c > 0] / np.sum(c)
    return -np.sum(p * np.log(p))


def mse(Y):
    return np.mean(np.square(Y - np.mean(Y)))


def gini_list(Y):
    n_samples = len(Y)

    # init
    rc = count(Y)
    lc = np.zeros_like(rc)
    ls = 0
    rs = np.sum(np.square(rc))

    scores = np.zeros(n_samples - 1)
    for i in range(n_samples - 1):
        # update
        k = Y[i]
        ls += 2 * lc[k] + 1
        lc[k] += 1
        rs -= 2 * rc[k] - 1
        rc[k] -= 1
        # score
        ln = i + 1
        rn = n_samples - ln
        scores[i] = 1 - (ls / ln + rs / rn) / n_samples
    return scores


def entropy_list(Y):
    n_samples = len(Y)

    # init
    rc = count(Y)
    lc = np.zeros_like(rc)

    scores = np.zeros(n_samples - 1)
    for i in range(n_samples - 1):
        # update
        k = Y[i]
        lc[k] += 1
        rc[k] -= 1
        # score
        ln = i + 1
        rn = n_samples - ln
        scores[i] = (ln * entropy(lc) + rn * entropy(rc)) / n_samples
    return scores


def mse_list(Y):
    n_samples = len(Y)

    # init
    ls = 0
    ls2 = 0
    rs = np.sum(Y)
    rs2 = np.sum(np.square(Y))

    scores = np.zeros(n_samples - 1)
    for i in range(n_samples - 1):
        # update
        y = Y[i]
        ls += y
        ls2 += y * y
        rs -= y
        rs2 -= y * y
        # score
        ln = i + 1
        rn = n_samples - ln
        scores[i] = (ls2 - ls * ls / ln + rs2 - rs * rs / rn) / n_samples
    return scores


criterion_func_dict = {
    'gini': lambda Y: gini(count(Y)),
    'entropy': lambda Y: entropy(count(Y)),
    'mse': mse
}

criterion_list_func_dict = {
    'gini': gini_list,
    'entropy': entropy_list,
    'mse': mse_list
}


##################################################
class Node:
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def depth(self):
        raise NotImplemented

    def walk(self, func):
        raise NotImplemented


class Inner(Node):
    def __init__(self, n_samples, gain, split_feature, split_value, left, right):
        super().__init__(n_samples)
        self.gain = gain
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right

    def depth(self):
        return max(self.left.depth(), self.right.depth()) + 1

    def walk(self, func):
        func(self)
        self.left.walk(func)
        self.right.walk(func)


class Leaf(Node):
    def __init__(self, n_samples, predict_value):
        super().__init__(n_samples)
        self.predict_value = predict_value

    def depth(self):
        return 0

    def walk(self, func):
        func(self)


class Tree:
    def __init__(self, criterion, max_depth=np.inf, alpha=1e-4):
        self.root = None
        self.n_features = None
        self.n_classes = None
        self.criterion = criterion
        self.criterion_func = criterion_func_dict[criterion]
        self.criterion_list_func = criterion_list_func_dict[criterion]
        self.max_depth = max_depth
        self.alpha = alpha

    def fit(self, X, Y):
        self.n_features = X.shape[1]
        self.n_classes = np.max(Y) + 1
        self.root = self.build_tree(X, Y, np.arange(X.shape[0]), 0)

    def predict(self, X):
        return np.array([self.predict_one(X[i]) for i in range(X.shape[0])])

    def build_tree(self, X, Y, mask, depth):
        ys = Y[mask]
        n_samples = len(mask)

        if depth >= self.max_depth:
            return self.build_leaf(ys)

        score_p = self.criterion_func(Y[mask])
        if score_p < self.alpha:
            return self.build_leaf(ys)

        score, feature, value = self.best_split(X, Y, mask)
        gain = score_p - score
        if score_p - score < self.alpha:
            return self.build_leaf(ys)

        on_left = X[mask, feature] < value
        left = self.build_tree(X, Y, mask[on_left], depth + 1)
        right = self.build_tree(X, Y, mask[~on_left], depth + 1)
        return Inner(n_samples, gain, feature, value, left, right)

    def build_leaf(self, Y):
        raise NotImplemented

    def best_split(self, X, Y, mask):
        best_score = np.inf
        best_feature = None
        best_value = None
        for feature in range(X.shape[1]):
            score, value = self.feature_best_split(X, Y, mask, feature)
            if best_score > score:
                best_score = score
                best_feature = feature
                best_value = value
        return best_score, best_feature, best_value

    def feature_best_split(self, X, Y, mask, feature):
        xs = X[mask, feature]
        ys = Y[mask]
        si = xs.argsort()
        xs = xs[si]
        ys = ys[si]
        scores = self.criterion_list_func(ys)
        best_score = np.inf
        best_value = None
        for i in range(xs.shape[0] - 1):
            if xs[i] < xs[i + 1] and best_score > scores[i]:
                best_score = scores[i]
                best_value = (xs[i] + xs[i + 1]) / 2
        return best_score, best_value

    def predict_one(self, x):
        p = self.root
        while isinstance(p, Inner):
            if x[p.split_feature] < p.split_value:
                p = p.left
            else:
                p = p.right
        return p.predict_value

    def get_feature_importances(self):
        importances = np.zeros(self.n_features)

        def f(node):
            if isinstance(node, Inner):
                importances[node.split_feature] += node.n_samples * node.gain

        self.root.walk(f)
        importances /= np.sum(importances)
        return importances


class TreeClassifier(Tree):
    def __init__(self, criterion='gini', max_depth=np.inf, alpha=1e-4):
        super().__init__(criterion, max_depth, alpha)

    def build_leaf(self, Y):
        return Leaf(len(Y), np.argmax(count(Y)))


class TreeRegressor(Tree):
    def __init__(self, criterion='mse', max_depth=np.inf, alpha=1e-4):
        super().__init__(criterion, max_depth, alpha)

    def build_leaf(self, Y):
        return Leaf(len(Y), np.mean(Y))
