import copy
import numpy as np

from .decision_tree import DecisionTreeClassifier

__all__ = ['AdaBoostClassifier']

TINY = np.finfo(np.float64).tiny


class AdaBoostClassifier:
    def __init__(self, base_estimator=None, n_estimators=50):
        if base_estimator is not None:
            self.base_estimator = base_estimator
        else:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.estimators_ = None
        self.estimator_errors_ = None
        self.estimator_weights_ = None
        self.feature_importances_ = None
        self.classes_ = None

    def fit(self, X, Y):
        self.classes_, Yi = np.unique(Y, return_inverse=True)

        n_samples, n_features = X.shape
        n_classes = self.classes_.shape[0]

        estimators = []
        estimator_errors = []
        estimator_weights = []
        weight = np.full(n_samples, 1 / n_samples)

        for i in range(self.n_estimators):
            estimator = copy.copy(self.base_estimator)
            estimator.fit(X, Yi, weight)
            incorrect = estimator.predict(X) != Yi

            if not np.any(incorrect):
                estimators.append(estimator)
                estimator_errors.append(0.0)
                estimator_weights.append(1.0)
                break

            error = np.sum(incorrect * weight)
            alpha = np.log((1 - error) / error) + np.log(n_classes - 1)
            if alpha <= 0:
                break

            estimators.append(estimator)
            estimator_errors.append(error)
            estimator_weights.append(alpha)

            weight *= np.exp(incorrect * alpha)
            weight /= np.sum(weight)

        self.estimators_ = estimators
        self.estimator_errors_ = np.array(estimator_errors)
        self.estimator_weights_ = np.array(estimator_weights)

        importances = np.zeros(n_features)
        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            importances += estimator.feature_importances_ * alpha
        self.feature_importances_ = importances / np.maximum(TINY, np.sum(importances))

    def predict(self, X):
        s = self.decision_function(X)
        return self.classes_[np.argmax(s, axis=1)]

    def decision_function(self, X):
        n_samples = X.shape[0]
        n_classes = self.classes_.shape[0]
        s = np.zeros((n_samples, n_classes))
        a = np.arange(n_samples)
        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            s[a, estimator.predict(X)] += alpha
        return s
