import numpy as np

__all__ = ['DecisionTreeClassifier', 'DecisionTreeRegressor']

TINY = np.finfo(np.float64).tiny


def gini(c):
    p = c / np.maximum(TINY, np.sum(c, axis=-1))[..., np.newaxis]
    return 1 - np.sum(np.square(p), axis=-1)


def entropy(c):
    p = c / np.maximum(TINY, np.sum(c, axis=-1))[..., np.newaxis]
    lp = np.log(p, out=np.zeros_like(p), where=p > 0)
    return -np.sum(p * lp, axis=-1)


def count(Y, weight, n_classes):
    n_samples = Y.shape[0]
    a = np.arange(n_samples)
    t = np.zeros((n_samples, n_classes))
    t[a, Y] = weight
    return t.sum(axis=0)


def count_all_split(Y, weight, n_classes):
    n_samples = Y.shape[0]
    a = np.arange(n_samples)
    t = np.zeros((n_samples, n_classes))
    t[a, Y] = weight
    c = np.add.accumulate(t, axis=0)
    lc = c[:-1]
    rc = c[-1] - c[:-1]
    return lc, rc


def general_all_split(Y, weight, n_classes, criterion):
    w = np.add.accumulate(np.broadcast_to(weight, Y.shape))
    lw = w[:-1]
    rw = w[-1] - w[:-1]
    lc, rc = count_all_split(Y, weight, n_classes)
    return (lw * criterion(lc) + rw * criterion(rc)) / w[-1]


def mse(Y, weight):
    w = np.sum(np.broadcast_to(weight, Y.shape))
    s = np.sum(Y * weight)
    s2 = np.sum(np.square(Y) * weight)
    return (s2 - np.square(s) / w) / w


def mse_all_split(Y, weight):
    w = np.add.accumulate(np.broadcast_to(weight, Y.shape))
    lw = w[:-1]
    rw = w[-1] - w[:-1]

    s = np.add.accumulate(Y * weight)
    ls = s[:-1]
    rs = s[-1] - s[:-1]

    s2 = np.add.accumulate(np.square(Y) * weight)
    ls2 = s2[:-1]
    rs2 = s2[-1] - s2[:-1]

    return (ls2 - np.square(ls) / lw + rs2 - np.square(rs) / rw) / w[-1]


impurity_func_dict = {
    'gini': lambda Y, weight: gini(count(Y, weight, np.max(Y) + 1)),
    'entropy': lambda Y, weight: entropy(count(Y, weight, np.max(Y) + 1)),
    'mse': mse
}

impurity_all_split_func_dict = {
    'gini': lambda Y, weight: general_all_split(Y, weight, np.max(Y) + 1, gini),
    'entropy': lambda Y, weight: general_all_split(Y, weight, np.max(Y) + 1, entropy),
    'mse': mse_all_split
}


class Inner:
    __slots__ = ['sum_weight', 'impurity_decrease', 'split_feature', 'split_value', 'left', 'right']

    def __init__(self, sum_weight, impurity_decrease, split_feature, split_value, left, right):
        self.sum_weight = sum_weight
        self.impurity_decrease = impurity_decrease
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


class Leaf:
    __slots__ = ['sum_weight', 'predict_value', 'predict_proba', 'indexes']

    def __init__(self, sum_weight, predict_value, predict_proba=None, indexes=None):
        self.sum_weight = sum_weight
        self.predict_value = predict_value
        self.predict_proba = predict_proba
        self.indexes = indexes

    def depth(self):
        return 0

    def walk(self, func):
        func(self)


class DecisionTreeBase:
    def __init__(self, criterion=None, splitter='best', max_features=None, max_depth=np.inf, min_impurity_decrease=0.0):
        """
        :param criterion: string
            Impurity criterion, "gini", "entropy" or "mse".

        :param splitter: string (default="best")
            How many features are used at splitting.
            - if "best",   all features
            - if "random", random subset of all features

        :param max_features: int, float, string, or None (default=None)
            Maximum number of features used at splitting. Used when splitter == "random".
            - if int,    max_features
            - if float,  int(max_features * n_features)
            - if "sqrt", int(sqrt(n_features))
            - if "log2", int(log2(n_features))

        :param max_depth: int or inf (default=inf)
            Maximum depth of the decision tree.

        :param min_impurity_decrease: float (default=0.0)
            Minimum decrease of impurity required at splitting.
        """

        self.criterion = criterion
        self.impurity_func = impurity_func_dict[criterion]
        self.impurity_all_split_func = impurity_all_split_func_dict[criterion]
        self.splitter = splitter
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features_ = None
        self.max_features_ = None
        self.root_ = None
        self.feature_importances_ = None

    def fit(self, X, Y, weight=None):
        n_samples, self.n_features_ = X.shape
        if weight is None:
            weight = np.full(n_samples, 1 / n_samples)
        else:
            weight = weight / np.sum(weight)
        self._compute_max_feature()
        self.root_ = self._build_tree(X, Y, weight, np.arange(X.shape[0]), 0)
        self.feature_importances_ = self._compute_feature_importances()

    def predict(self, X):
        raise NotImplementedError

    def _compute_max_feature(self):
        if self.splitter == 'random':
            if isinstance(self.max_features, int) and self.max_features >= 1:
                n = self.max_features
            elif isinstance(self.max_features, float) and 0 < self.max_features <= 1:
                n = self.max_features * self.n_features_
            elif self.max_features == 'sqrt':
                n = np.sqrt(self.n_features_)
            elif self.max_features == 'log2':
                n = np.log2(self.n_features_)
            else:
                raise ValueError('max_features')
            self.max_features_ = np.clip(int(n), 1, self.n_features_)
        elif self.splitter != 'best':
            raise ValueError('splitter')

    def _build_tree(self, X, Y, weight, indexes, depth):
        ys = Y[indexes]
        ws = weight[indexes]
        sum_weight = np.sum(ws)

        if depth >= self.max_depth or indexes.shape[0] <= 1:
            return self._build_leaf(ys, ws, indexes)

        impurity_p = self.impurity_func(ys, ws)
        if impurity_p * sum_weight <= self.min_impurity_decrease:
            return self._build_leaf(ys, ws, indexes)

        split_feature, split_value, impurity = self._best_split(X, Y, weight, indexes)
        impurity_decrease = impurity_p - impurity
        if impurity_decrease * sum_weight < self.min_impurity_decrease:
            return self._build_leaf(ys, ws, indexes)

        on_left = X[indexes, split_feature] < split_value
        left = self._build_tree(X, Y, weight, indexes[on_left], depth + 1)
        right = self._build_tree(X, Y, weight, indexes[~on_left], depth + 1)

        return Inner(sum_weight, impurity_decrease, split_feature, split_value, left, right)

    def _build_leaf(self, ys, ws, indexes):
        raise NotImplementedError

    def _best_split(self, X, Y, weight, indexes):
        if self.splitter == 'best':
            return self._best_split_fs(X, Y, weight, indexes, range(self.n_features_))
        else:
            fs = np.random.choice(self.n_features_, size=self.max_features_, replace=False)
            split_feature, split_value, impurity = self._best_split_fs(X, Y, weight, indexes, fs)
            if impurity == np.inf:
                s = set(fs)
                fr = np.array([f for f in range(self.n_features_) if f not in s])
                np.random.shuffle(fr)
                for feature in fr:
                    split_value_c, impurity_c = self._best_split_f(X, Y, weight, indexes, feature)
                    if impurity_c < np.inf:
                        split_feature = feature
                        split_value = split_value_c
                        impurity = impurity_c
                        break
            return split_feature, split_value, impurity

    def _best_split_fs(self, X, Y, weight, indexes, features):
        sv_im = np.array([self._best_split_f(X, Y, weight, indexes, feature)
                          for feature in features])
        i = np.argmin(sv_im[:, 1])
        split_feature = features[i]
        split_value = sv_im[i, 0]
        impurity = sv_im[i, 1]
        return split_feature, split_value, impurity

    def _best_split_f(self, X, Y, weight, indexes, feature):
        a = indexes[np.argsort(X[indexes, feature])]
        xs = X[a, feature]
        ys = Y[a]
        ws = weight[a]

        impurity_list = self.impurity_all_split_func(ys, ws)
        valid_impurity_list = np.where(xs[:-1] < xs[1:], impurity_list, np.inf)

        i = np.argmin(valid_impurity_list)
        split_value = (xs[i] + xs[i + 1]) / 2
        impurity = valid_impurity_list[i]

        return split_value, impurity

    def _find_leaf(self, x):
        p = self.root_
        while isinstance(p, Inner):
            if x[p.split_feature] < p.split_value:
                p = p.left
            else:
                p = p.right
        return p

    def _compute_feature_importances(self):
        def f(node):
            if isinstance(node, Inner):
                importances[node.split_feature] += node.sum_weight * node.impurity_decrease

        importances = np.zeros(self.n_features_)
        self.root_.walk(f)
        return importances / np.maximum(TINY, np.sum(importances))


class DecisionTreeClassifier(DecisionTreeBase):
    def __init__(self, criterion='gini', splitter='best', max_features=None, max_depth=np.inf,
                 min_impurity_decrease=0.0):
        """
        :param criterion: string (default="gini")
            Impurity criterion, "gini" or "entropy".

        :param splitter: string (default="best")
            How many features are used at splitting.
            - if "best",   all features
            - if "random", random subset of all features

        :param max_features: int, float, string, or None (default=None)
            Maximum number of features used at splitting. Used when splitter == "random".
            - if int,    max_features
            - if float,  int(max_features * n_features)
            - if "sqrt", int(sqrt(n_features))
            - if "log2", int(log2(n_features))

        :param max_depth: int or inf (default=inf)
            Maximum depth of the decision tree.

        :param min_impurity_decrease: float (default=0.0)
            Minimum decrease of impurity required at splitting.
        """

        super().__init__(criterion, splitter, max_features, max_depth, min_impurity_decrease)
        self.classes_ = None

    def fit(self, X, Y, weight=None):
        self.classes_, Yi = np.unique(Y, return_inverse=True)
        super().fit(X, Yi, weight)

    def predict(self, X):
        a = [self._find_leaf(X[i]).predict_value for i in range(X.shape[0])]
        return self.classes_[a]

    def predict_proba(self, X):
        return np.stack([self._find_leaf(X[i]).predict_proba for i in range(X.shape[0])])

    def _build_leaf(self, ys, ws, indexes):
        c = count(ys, ws, self.classes_.shape[0])
        return Leaf(np.sum(ws), np.argmax(c), predict_proba=c / np.sum(c))


class DecisionTreeRegressor(DecisionTreeBase):
    def __init__(self, criterion='mse', splitter='best', max_features=None, max_depth=np.inf,
                 min_impurity_decrease=0.0):
        """
        :param criterion: string (default="mse")
            Impurity criterion, "mse".

        :param splitter: string (default="best")
            How many features are used at splitting.
            - if "best",   all features
            - if "random", random subset of all features

        :param max_features: int, float, string, or None (default=None)
            Maximum number of features used at splitting. Used when splitter == "random".
            - if int,    max_features
            - if float,  int(max_features * n_features)
            - if "sqrt", int(sqrt(n_features))
            - if "log2", int(log2(n_features))

        :param max_depth: int or inf (default=inf)
            Maximum depth of the decision tree.

        :param min_impurity_decrease: float (default=0.0)
            Minimum decrease of impurity required at splitting.
        """

        super().__init__(criterion, splitter, max_features, max_depth, min_impurity_decrease)

    def predict(self, X):
        return np.array([self._find_leaf(X[i]).predict_value for i in range(X.shape[0])])

    def _build_leaf(self, ys, ws, indexes):
        sum_weight = np.sum(ws)
        return Leaf(sum_weight, np.sum(ys * ws) / sum_weight, indexes=indexes)
