import numpy as np

from distance import l2_distance, pairwise_l2_distance


class KMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        cluster_centers = None
        labels = None
        inertia = np.inf
        for _ in range(self.n_init):
            c, l, i = self._iter(X)
            if inertia > i:
                cluster_centers = c
                labels = l
                inertia = i
        self.cluster_centers_ = cluster_centers
        self.labels_ = labels
        self.inertia_ = inertia

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        d = pairwise_l2_distance(X, self.cluster_centers_, square=True)
        return np.argmin(d, axis=1)

    def _iter(self, X):
        n_samples = X.shape[0]
        a = range(n_samples)

        centers = self._init_centers(X)
        labels = None
        inertia = np.inf
        for _ in range(self.max_iter):
            d = pairwise_l2_distance(X, centers, square=True)
            labels = np.argmin(d, axis=1)

            new_inertia = np.sum(d[a, labels])
            if new_inertia - inertia >= - self.tol:
                break

            new_centers = np.zeros_like(centers)
            for i in range(self.n_clusters):
                mask = labels == i
                if np.any(mask):
                    new_centers[i] = np.mean(X[mask], axis=0)

            centers = new_centers
            inertia = new_inertia

        return centers, labels, inertia

    def _init_centers(self, X):
        n_samples = X.shape[0]
        indexes = []
        d = np.full(n_samples, np.inf)
        for i in range(self.n_clusters):
            a = np.argsort(d)
            index = a[np.random.randint((n_samples + i) // 2, n_samples)]
            indexes.append(index)
            d = np.minimum(d, l2_distance(X, X[index], square=True))
            d[index] = -1
        return X[indexes]
