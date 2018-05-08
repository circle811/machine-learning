import numpy as np

from .decision_tree import Leaf, DecisionTreeRegressor

__all__ = ['GradientBoostingClassifier', 'GradientBoostingRegressor']

TINY = np.finfo(np.float64).tiny

init_func_dict = {
    'deviance': None,
    'ls': np.mean,
    'lad': np.median,
    'huber': np.median
}

neg_gradient_func_dict = {
    'deviance': None,
    'ls': lambda y_true, y_pred: y_true - y_pred,
    'lad': lambda y_true, y_pred: np.sign(y_true - y_pred)
}

predict_value_func_dict = {
    'deviance': None,
    'ls': np.mean,
    'lad': np.median
}


def huber(y_true, y_pred, delta):
    d = np.abs(y_true - y_pred)
    return np.where(d <= delta, 0.5 * np.square(d), delta * (d - 0.5 * delta))


def huber_neg_gradient(y_true, y_pred, delta):
    return np.clip(y_true - y_pred, -delta, delta)


def huber_predict_value(y, delta):
    p0 = np.median(y)
    return p0 + np.mean(huber_neg_gradient(y, p0, delta))


def adjust_tree(predict_value_func, tree, Y):
    def f(node):
        if isinstance(node, Leaf):
            node.predict_value = predict_value_func(Y[node.indexes])

    tree.root_.walk(f)


class GradientBoostingBase:
    pass


class GradientBoostingClassifier(GradientBoostingBase):
    pass


class GradientBoostingRegressor(GradientBoostingBase):
    def __init__(self, n_estimators=100, learning_rate=0.1, loss='ls', alpha=0.9, criterion='mse', max_depth=3,
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
        estimators = []
        y_pred = np.full(n_samples, init)

        for i in range(self.n_estimators):
            if self.loss == 'huber':
                delta = np.percentile(np.abs(Y - y_pred), self.alpha * 100)
                neg_grad = huber_neg_gradient(Y, y_pred, delta)
                predict_value_func = lambda y: huber_predict_value(y, delta)
            else:
                neg_grad = neg_gradient_func_dict[self.loss](Y, y_pred)
                predict_value_func = predict_value_func_dict[self.loss]
            estimator = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth,
                                              min_impurity_decrease=self.min_impurity_decrease)
            estimator.fit(X, neg_grad)
            adjust_tree(predict_value_func, estimator, Y - y_pred)
            estimators.append(estimator)
            y_pred += self.learning_rate * estimator.predict(X)

        self.init_ = init
        self.estimators_ = np.array(estimators)

    def predict(self, X):
        s = np.full(X.shape[0], self.init_)
        for estimator in self.estimators_:
            s += self.learning_rate * estimator.predict(X)
        return s
