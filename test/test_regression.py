import os
import sys
import time
import numpy as np

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:
    sys.path.append(path)

from machine_learning.algorithm import optimizer
from machine_learning.supervised import (
    knn, linear, decision_tree, random_forest, neural_network)
from machine_learning.utils import load_mnist

x_train, y_train = load_mnist.load_mnist('train')
x_test, y_test = load_mnist.load_mnist('test')
y_train = y_train - 4.5
y_test = y_test - 4.5


def test(class_, params, train_size, test_size):
    print('class_={}, params={}, train_size={}, test_size={}'
          .format(class_.__name__, params, train_size, test_size))

    x = x_train[:train_size]
    y = y_train[:train_size]
    xt = x_test[:test_size]
    yt = y_test[:test_size]

    # fit
    t0 = time.time()
    c = class_(**params)
    c.fit(x, y)
    t1 = time.time()

    # predict
    yp = c.predict(xt)
    ev = 1 - np.var(yt - yp) / np.var(yt)
    r2 = 1 - np.sum(np.square(yt - yp)) / np.sum(np.square(yt - np.mean(yt)))
    mse = np.mean(np.square(yt - yp))
    t2 = time.time()

    print('explained_variance={:.6f}, r2_score={:.6f}, mean_squared_error={:.6f}, fit={:.3f}s, predict={:.3f}s'
          .format(ev, r2, mse, t1 - t0, t2 - t1, ))
    print()


class_params_list = [
    (knn.KNNRegressor, dict(n_neighbors=5, metric='l1', algorithm='kd_tree')),
    (knn.KNNRegressor, dict(n_neighbors=5, metric='l1', algorithm='brute')),
    (knn.KNNRegressor, dict(n_neighbors=5, metric='l2', algorithm='kd_tree')),
    (knn.KNNRegressor, dict(n_neighbors=5, metric='l2', algorithm='brute')),
    (knn.KNNRegressor, dict(n_neighbors=5, metric='l2_square', algorithm='kd_tree')),
    (knn.KNNRegressor, dict(n_neighbors=5, metric='l2_square', algorithm='brute')),
    (knn.KNNRegressor, dict(n_neighbors=5, metric='linf', algorithm='kd_tree')),
    (knn.KNNRegressor, dict(n_neighbors=5, metric='linf', algorithm='brute')),
    (knn.KNNRegressor, dict(n_neighbors=10, metric='l2_square', algorithm='kd_tree')),
    (knn.KNNRegressor, dict(n_neighbors=10, metric='l2_square', algorithm='brute')),

    (linear.LinearRegression, dict(alpha=0.0)),
    (linear.LinearRegression, dict(alpha=0.0001, l1_ratio=0.0)),
    (linear.LinearRegression, dict(alpha=0.0001, l1_ratio=0.5)),
    (linear.LinearRegression, dict(alpha=0.0001, l1_ratio=1.0)),

    (decision_tree.DecisionTreeRegressor, dict(criterion='mse', splitter='best', min_impurity_decrease=0.0)),
    (decision_tree.DecisionTreeRegressor, dict(criterion='mse', splitter='best', min_impurity_decrease=1e-5)),

    (decision_tree.DecisionTreeRegressor, dict(criterion='mse', splitter='random', max_features='log2')),
    (decision_tree.DecisionTreeRegressor, dict(criterion='mse', splitter='random', max_features='sqrt')),
    (decision_tree.DecisionTreeRegressor, dict(criterion='mse', splitter='random', max_features=0.2)),
    (decision_tree.DecisionTreeRegressor, dict(criterion='mse', splitter='random', max_features=30)),

    (random_forest.RandomForestRegressor, dict(n_estimators=10, max_features='log2')),
    (random_forest.RandomForestRegressor, dict(n_estimators=10, max_features='sqrt')),
    (random_forest.RandomForestRegressor, dict(n_estimators=20, max_features='log2')),
    (random_forest.RandomForestRegressor, dict(n_estimators=20, max_features='sqrt')),

    (neural_network.NeuralNetworkRegressor, dict(hidden_layer_sizes=(200,), optimizer=optimizer.AdamOptimizer())),
    (neural_network.NeuralNetworkRegressor, dict(hidden_layer_sizes=(200,), optimizer=optimizer.SGDOptimizer())),
    (neural_network.NeuralNetworkRegressor, dict(hidden_layer_sizes=(100, 100), optimizer=optimizer.AdamOptimizer())),
    (neural_network.NeuralNetworkRegressor, dict(hidden_layer_sizes=(100, 100), optimizer=optimizer.SGDOptimizer())),
]

print('start test')
for class_, params in class_params_list:
    test(class_, params, 800, 1000)
