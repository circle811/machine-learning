import numpy as np

from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = ['RandomForestClassifier', 'RandomForestRegressor']

TINY = np.finfo(np.float64).tiny


class RandomForestBase:
    DecisionTree = None

    def __init__(self, n_estimators=10, criterion=None, max_features='sqrt', max_depth=np.inf,
                 min_impurity_decrease=0.0):
        """
        :param n_estimators: int (default=10)
            Maximum number of decision trees.

        :param criterion: string
            Impurity criterion, "gini", "entropy" or "mse".

        :param max_features: int, float or string (default="sqrt")
            Maximum number of features used at splitting.
            - if int,    max_features
            - if float,  int(max_features * n_features)
            - if "sqrt", int(sqrt(n_features))
            - if "log2", int(log2(n_features))

        :param max_depth: int or inf (default=inf)
            Maximum depth of the decision tree.

        :param min_impurity_decrease: float (default=0.0)
            Minimum decrease of impurity required at splitting.
        """

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.estimators_ = None
        self.feature_importances_ = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape

        estimators = []
        for i in range(self.n_estimators):
            estimator = self.DecisionTree(criterion=self.criterion, splitter='random', max_features=self.max_features,
                                          max_depth=self.max_depth, min_impurity_decrease=self.min_impurity_decrease)
            a = np.random.choice(n_samples, size=n_samples, replace=True)
            a.sort()
            estimator.fit(X[a], Y[a])
            estimators.append(estimator)

        self.estimators_ = np.array(estimators)

        importances = np.zeros(n_features)
        for estimator in self.estimators_:
            importances += estimator.feature_importances_
        self.feature_importances_ = importances / np.maximum(TINY, np.sum(importances))

    def predict(self, X):
        raise NotImplementedError


class RandomForestClassifier(RandomForestBase):
    DecisionTree = DecisionTreeClassifier

    def __init__(self, n_estimators=10, criterion='gini', max_features='sqrt', max_depth=np.inf,
                 min_impurity_decrease=0.0):
        """
        :param n_estimators: int (default=10)
            Maximum number of decision trees.

        :param criterion: string (default="gini")
            Impurity criterion, "gini" or "entropy".

        :param max_features: int, float or string (default="sqrt")
            Maximum number of features used at splitting.
            - if int,    max_features
            - if float,  int(max_features * n_features)
            - if "sqrt", int(sqrt(n_features))
            - if "log2", int(log2(n_features))

        :param max_depth: int or inf (default=inf)
            Maximum depth of the decision tree.

        :param min_impurity_decrease: float (default=0.0)
            Minimum decrease of impurity required at splitting.
        """

        super().__init__(n_estimators, criterion, max_features, max_depth, min_impurity_decrease)
        self.classes_ = None

    def fit(self, X, Y):
        self.classes_, Yi = np.unique(Y, return_inverse=True)
        super().fit(X, Yi)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]

    def predict_proba(self, X):
        s = np.zeros((X.shape[0], self.classes_.shape[0]))
        for estimator in self.estimators_:
            s += estimator.predict_proba(X)
        return s / self.n_estimators


class RandomForestRegressor(RandomForestBase):
    DecisionTree = DecisionTreeRegressor

    def __init__(self, n_estimators=10, criterion='mse', max_features='sqrt', max_depth=np.inf,
                 min_impurity_decrease=0.0):
        """
        :param n_estimators: int (default=10)
            Maximum number of decision trees.

        :param criterion: string (default="mse")
            Impurity criterion, "mse".

        :param max_features: int, float or string (default="sqrt")
            Maximum number of features used at splitting.
            - if int,    max_features
            - if float,  int(max_features * n_features)
            - if "sqrt", int(sqrt(n_features))
            - if "log2", int(log2(n_features))

        :param max_depth: int or inf (default=inf)
            Maximum depth of the decision tree.

        :param min_impurity_decrease: float (default=0.0)
            Minimum decrease of impurity required at splitting.
        """

        super().__init__(n_estimators, criterion, max_features, max_depth, min_impurity_decrease)

    def predict(self, X):
        s = np.zeros(X.shape[0])
        for estimator in self.estimators_:
            s += estimator.predict(X)
        return s / self.n_estimators
