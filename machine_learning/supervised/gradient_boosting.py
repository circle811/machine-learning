import numpy as np

from .decision_tree import Leaf, DecisionTreeRegressor

__all__ = ['GradientBoostingClassifier', 'GradientBoostingRegressor']

TINY = np.finfo(np.float64).tiny


def deviance_init(y):
    p = y.sum(axis=0) / y.shape[0]
    lp = np.log(p)
    return lp - np.mean(lp)


def deviance_d_r(y, s):
    p = np.exp(s - np.max(s, axis=1)[:, np.newaxis])
    p /= np.sum(p, axis=1)[:, np.newaxis]
    d = y - p
    r = d
    return d, r


def deviance_predict_value(d, n_classes):
    abs_d = np.abs(d)
    return (n_classes - 1) * np.sum(d) / (n_classes * np.sum(abs_d * (1 - abs_d)))


def ls_d_r(y, s):
    d = y - s
    r = d
    return d, r


def lad_d_r(y, s):
    d = y - s
    r = np.sign(d)
    return d, r


def huber_d_r(y, s, delta):
    d = y - s
    r = np.clip(d, -delta, delta)
    return d, r


def huber_predict_value(d, delta):
    p0 = np.median(d)
    return p0 + np.mean(np.clip(d - p0, -delta, delta))


init_func_dict = {
    'deviance': deviance_init,
    'ls': lambda y: np.mean(y)[np.newaxis],
    'lad': lambda y: np.median(y)[np.newaxis],
    'huber': lambda y: np.median(y)[np.newaxis]
}

d_r_func_dict = {
    'deviance': deviance_d_r,
    'ls': ls_d_r,
    'lad': lad_d_r
}

predict_value_func_dict = {
    'ls': np.mean,
    'lad': np.median
}


def adjust_tree(predict_value_func, tree, Y):
    def f(node):
        if isinstance(node, Leaf):
            node.predict_value = predict_value_func(Y[node.indexes])

    tree.root_.walk(f)


class GradientBoostingBase:
    def __init__(self, n_estimators=100, learning_rate=0.1, loss=None, alpha=None, criterion='mse', max_depth=3,
                 min_impurity_decrease=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.alpha = alpha
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init_ = None
        self.estimators_ = None
        self.feature_importances_ = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape

        init = init_func_dict[self.loss](Y)
        n_output = init.shape[0]
        estimators = np.zeros((self.n_estimators, n_output), dtype=np.object_)
        s = np.tile(init, (n_samples, 1))

        if self.loss == 'deviance':
            predict_value_func = lambda d: deviance_predict_value(d, n_output)
        elif self.loss == 'huber':
            predict_value_func = None
        else:
            predict_value_func = predict_value_func_dict[self.loss]

        for i in range(self.n_estimators):
            if self.loss == 'huber':
                delta = np.percentile(np.abs(Y - s), self.alpha * 100)
                d, r = huber_d_r(Y, s, delta)
                predict_value_func = lambda d: huber_predict_value(d, delta)
            else:
                d, r = d_r_func_dict[self.loss](Y, s)

            for j in range(n_output):
                estimator = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth,
                                                  min_impurity_decrease=self.min_impurity_decrease)
                estimator.fit(X, r[:, j])
                adjust_tree(predict_value_func, estimator, d[:, j])
                estimators[i, j] = estimator
                s[:, j] += self.learning_rate * estimators[i, j].predict(X)

        self.init_ = init
        self.estimators_ = estimators

        importances = np.zeros(n_features)
        for i in range(self.n_estimators):
            for j in range(n_output):
                importances += self.estimators_[i, j].feature_importances_
        self.feature_importances_ = importances / np.maximum(TINY, np.sum(importances))

    def predict(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        n_samples = X.shape[0]
        n_estimators, n_output = self.estimators_.shape
        s = np.tile(self.init_, (n_samples, 1))
        for i in range(n_estimators):
            for j in range(n_output):
                s[:, j] += self.learning_rate * self.estimators_[i, j].predict(X)
        return s


class GradientBoostingClassifier(GradientBoostingBase):
    def __init__(self, n_estimators=100, learning_rate=0.1, loss='deviance', criterion='mse', max_depth=3,
                 min_impurity_decrease=0.0):
        super().__init__(n_estimators, learning_rate, loss, None, criterion, max_depth, min_impurity_decrease)
        self.classes_ = None

    def fit(self, X, Y):
        n_samples = X.shape[0]
        self.classes_, Yi = np.unique(Y, return_inverse=True)
        Ye = np.zeros((n_samples, self.classes_.shape[0]))
        Ye[np.arange(n_samples), Yi] += 1
        super().fit(X, Ye)

    def predict(self, X):
        s = self.decision_function(X)
        return self.classes_[np.argmax(s, axis=1)]

    def predict_proba(self, X):
        s = self.decision_function(X)
        p = np.exp(s - np.max(s, axis=1)[:, np.newaxis])
        p /= np.sum(p, axis=1)[:, np.newaxis]
        return p


class GradientBoostingRegressor(GradientBoostingBase):
    def __init__(self, n_estimators=100, learning_rate=0.1, loss='ls', alpha=0.9, criterion='mse', max_depth=3,
                 min_impurity_decrease=0.0):
        super().__init__(n_estimators, learning_rate, loss, alpha, criterion, max_depth, min_impurity_decrease)

    def fit(self, X, Y):
        super().fit(X, Y[:, np.newaxis])

    def predict(self, X):
        return self.decision_function(X)[:, 0]
