import numpy as np

__all__ = ['BernoulliNB', 'MultinomialNB', 'GaussianNB']


class NBBase:
    def __init__(self):
        self.classes_ = None
        self.class_log_prior_ = None

    def fit(self, X, Y):
        self.classes_, Yi, c = np.unique(Y, return_inverse=True, return_counts=True)
        self.class_log_prior_ = np.log(c / Y.shape[0])
        self._estimate(X, Yi)

    def predict(self, X):
        s = self.decision_function(X)
        return self.classes_[np.argmax(s, axis=1)]

    def predict_proba(self, X):
        s = self.decision_function(X)
        s -= np.max(s, axis=1)[:, np.newaxis]
        p = np.exp(s)
        p /= np.sum(p, axis=1)[:, np.newaxis]
        return p

    def _estimate(self, X, Yi):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError


class BernoulliNB(NBBase):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.feature_log_prob_ = None
        self.feature_log_neg_prob_ = None

    def _estimate(self, X, Yi):
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        self.feature_log_neg_prob_ = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            Xi = X[Yi == i]
            c = np.count_nonzero(Xi, axis=0)
            p = (c + self.alpha) / (Xi.shape[0] + 2 * self.alpha)
            self.feature_log_prob_[i] = np.log(p)
            self.feature_log_neg_prob_[i] = np.log(1 - p)

    def decision_function(self, X):
        return (self.class_log_prior_ +
                (X != 0) @ (self.feature_log_prob_ - self.feature_log_neg_prob_).T +
                np.sum(self.feature_log_neg_prob_, axis=1))


class MultinomialNB(NBBase):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.feature_log_prob_ = None

    def _estimate(self, X, Yi):
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            Xi = X[Yi == i]
            c = np.sum(Xi, axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(c / np.sum(c))

    def decision_function(self, X):
        return self.class_log_prior_ + X @ self.feature_log_prob_.T


class GaussianNB(NBBase):
    def __init__(self):
        super().__init__()
        self.theta_ = None
        self.sigma_ = None

    def _estimate(self, X, Yi):
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        for i in range(n_classes):
            Xi = X[Yi == i]
            th = np.mean(Xi, axis=0)
            self.theta_[i] = th
            self.sigma_[i] = np.mean(np.square(Xi - th), axis=0)
        self.sigma_ += 1e-9 * np.maximum(0.1, np.max(self.sigma_))

    def decision_function(self, X):
        return (self.class_log_prior_ +
                -0.5 * np.sum(np.log(self.sigma_) +
                              np.square(X[:, np.newaxis, :] - self.theta_) / self.sigma_,
                              axis=2))
