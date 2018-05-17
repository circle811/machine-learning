import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:
    sys.path.append(path)

from machine_learning.reduction import pca, isomap, tsne
from machine_learning.utils import load_mnist

x_train, y_train = load_mnist.load_mnist('train')
n_classes = 5

p = pca.PCA()
x_train_subset_pca = p.fit_transform(x_train[y_train < n_classes][:10000])
y_train_subset = y_train[y_train < n_classes][:10000]


def plot(xt, y, title):
    _, ax = plt.subplots()
    ax.set_title(title)
    for i in range(n_classes):
        xs = xt[y == i]
        ax.scatter(xs[:, 0], xs[:, 1])


def test(class_, params, train_size, n_features):
    print('class_={}, params={}, train_size={}, n_features={}'
          .format(class_.__name__, params, train_size, n_features))

    x = x_train_subset_pca[:train_size, :n_features]
    y = y_train_subset[:train_size]

    t0 = time.time()
    c = class_(**params)
    xt = c.fit_transform(x)
    t1 = time.time()
    plot(xt, y, '{} {}'.format(class_.__name__, params))
    print('fit_transform={:.3f}s'.format(t1 - t0))
    print()

    return c


class_params_list = [
    (pca.PCA, dict(n_components=2)),
    (pca.KernelPCA, dict(n_components=2, kernel='rbf', gamma=0.0001)),
    (isomap.Isomap, dict(n_components=2, n_neighbors=10, algorithm='kd_tree')),
    (tsne.TSNE, dict(n_components=2)),
]

print('start test')
l = []
for class_, params in class_params_list:
    l.append(test(class_, params, 1000, 50))
plt.show()
