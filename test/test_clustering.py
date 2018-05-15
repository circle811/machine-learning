import os
import sys
import time
import numpy as np

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:
    sys.path.append(path)

from machine_learning.clustering import kmeans, gaussian_mixture, agglomerative, dbscan
from machine_learning.reduction import pca
from machine_learning.utils import load_mnist, metrics

x_train, y_train = load_mnist.load_mnist('train')
n_classes = 10

p = pca.PCA()
x_train_pca = p.fit_transform(x_train[:10000])


def print_score(x, y, yp):
    n_labels = np.unique(yp).shape[0]
    ar = metrics.adjusted_rand_score(y, yp)
    fm = metrics.fowlkes_mallows_score(y, yp)
    nmi = metrics.normalized_mutual_info_score(y, yp)
    v = metrics.v_measure_score(y, yp)
    print('n_labels={}, adjusted_rand={:.6f}, fowlkes_mallows={:.6f}, normalized_mutual_info={:.6f}, v_measure={:.6f}'
          .format(n_labels, ar, fm, nmi, v, ))
    if n_labels > 1:
        s = metrics.silhouette_score(x, y)
        s_p = metrics.silhouette_score(x, yp)
        ch = metrics.calinski_harabaz_score(x, y)
        ch_p = metrics.calinski_harabaz_score(x, yp)
        print(
            'silhouette_true={:.6f}, silhouette_pred={:.6f}, calinski_harabaz_true={:.6f}, calinski_harabaz_pred={:.6f}'
            .format(s, s_p, ch, ch_p))


def test(class_, params, train_size, n_features):
    print('class_={}, params={}, train_size={}, n_features={}'
          .format(class_.__name__, params, train_size, n_features))

    x = x_train_pca[:train_size, :n_features]
    y = y_train[:train_size]

    c = None
    c1 = None

    # fit predict
    if hasattr(class_, 'predict'):
        t0 = time.time()
        c = class_(**params)
        c.fit(x)
        yp = c.predict(x)
        t1 = time.time()
        print('fit&predict={:.3f}s'.format(t1 - t0))
        print_score(x, y, yp)

    # fit_predict
    if hasattr(class_, 'fit_predict'):
        t2 = time.time()
        c1 = class_(**params)
        yp1 = c1.fit_predict(x)
        t3 = time.time()
        print('fit_predict={:.3f}s'.format(t3 - t2))
        print_score(x, y, yp1)

    print()

    return c, c1


class_params_list = [
    (kmeans.KMeans, dict(n_clusters=n_classes)),

    (gaussian_mixture.GaussianMixture, dict(n_components=n_classes)),

    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='ward')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='single', affinity='l1')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='single', affinity='l2')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='single', affinity='l2_square')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='single', affinity='linf')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='complete', affinity='l1')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='complete', affinity='l2')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='complete', affinity='l2_square')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='complete', affinity='linf')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='average', affinity='l1')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='average', affinity='l2')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='average', affinity='l2_square')),
    (agglomerative.AgglomerativeClustering, dict(n_clusters=n_classes, linkage='average', affinity='linf')),

    (dbscan.DBSCAN, dict(metric='l2', min_samples=5, eps=3.9, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=5, eps=3.9, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=5, eps=4.0, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=5, eps=4.0, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=5, eps=4.1, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=5, eps=4.1, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=5, eps=4.2, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=5, eps=4.2, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=10, eps=3.9, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=10, eps=3.9, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=10, eps=4.0, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=10, eps=4.0, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=10, eps=4.1, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=10, eps=4.1, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=10, eps=4.2, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l2', min_samples=10, eps=4.2, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l1', min_samples=5, eps=21.0, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l1', min_samples=5, eps=21.0, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l1', min_samples=5, eps=22.0, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l1', min_samples=5, eps=22.0, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l1', min_samples=5, eps=23.0, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l1', min_samples=5, eps=23.0, algorithm='brute')),
    (dbscan.DBSCAN, dict(metric='l1', min_samples=5, eps=24.0, algorithm='kd_tree')),
    (dbscan.DBSCAN, dict(metric='l1', min_samples=5, eps=24.0, algorithm='brute')),
]

print('start test')
l = []
for class_, params in class_params_list:
    l.append(test(class_, params, 1000, 50))
