import os
import sys
import time
import numpy as np

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if path not in sys.path:
    sys.path.append(path)

from machine_learning.algorithm import optimizer
from machine_learning.supervised import (
    knn, naive_bayes, linear, svm, decision_tree, adaboost, gradient_boosting, random_forest, neural_network)
from machine_learning.utils import load_mnist

x_train, y_train = load_mnist.load_mnist('train')
x_test, y_test = load_mnist.load_mnist('test')
y_train = y_train * 10 + 100
y_test = y_test * 10 + 100


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
    accuracy = np.mean(yp == yt)
    t2 = time.time()

    if hasattr(c, 'predict_proba'):
        # predict_proba
        _, ytt = np.unique(yt, return_inverse=True)
        ypp = c.predict_proba(xt)
        p = ypp[np.arange(ytt.shape[0]), ytt]
        logloss = -np.mean(np.log(np.maximum(1e-15, p)))
        t3 = time.time()

        print('accuracy={:.4f}, logloss={:.4f}, fit={:.3f}s, predict={:.3f}s, predict_proba={:.3f}s'
              .format(accuracy, logloss, t1 - t0, t2 - t1, t3 - t2))
    else:
        print('accuracy={:.4f}, fit={:.3f}s, predict={:.3f}s'
              .format(accuracy, t1 - t0, t2 - t1))
    print()

    return c


class_params_list = [
    (knn.KNNClassifier, dict(n_neighbors=5, metric='l1', algorithm='kd_tree')),
    (knn.KNNClassifier, dict(n_neighbors=5, metric='l1', algorithm='brute')),
    (knn.KNNClassifier, dict(n_neighbors=5, metric='l2', algorithm='kd_tree')),
    (knn.KNNClassifier, dict(n_neighbors=5, metric='l2', algorithm='brute')),
    (knn.KNNClassifier, dict(n_neighbors=5, metric='l2_square', algorithm='kd_tree')),
    (knn.KNNClassifier, dict(n_neighbors=5, metric='l2_square', algorithm='brute')),
    (knn.KNNClassifier, dict(n_neighbors=5, metric='linf', algorithm='kd_tree')),
    (knn.KNNClassifier, dict(n_neighbors=5, metric='linf', algorithm='brute')),
    (knn.KNNClassifier, dict(n_neighbors=10, metric='l2_square', algorithm='kd_tree')),
    (knn.KNNClassifier, dict(n_neighbors=10, metric='l2_square', algorithm='brute')),

    (naive_bayes.BernoulliNB, dict(alpha=0.5)),
    (naive_bayes.BernoulliNB, dict(alpha=1.0)),
    (naive_bayes.MultinomialNB, dict(alpha=0.5)),
    (naive_bayes.MultinomialNB, dict(alpha=1.0)),
    (naive_bayes.GaussianNB, dict()),

    (linear.LogisticRegression, dict(alpha=0.01, l1_ratio=0.0, optimizer=optimizer.GD())),
    (linear.LogisticRegression, dict(alpha=0.01, l1_ratio=0.0, optimizer=optimizer.MomentumGD(nesterovs=False))),
    (linear.LogisticRegression, dict(alpha=0.01, l1_ratio=0.0, optimizer=optimizer.MomentumGD(nesterovs=True))),
    (linear.LogisticRegression, dict(alpha=0.01, l1_ratio=0.0, optimizer=optimizer.SGD())),
    (linear.LogisticRegression, dict(alpha=0.01, l1_ratio=0.0, optimizer=optimizer.MomentumSGD(nesterovs=False))),
    (linear.LogisticRegression, dict(alpha=0.01, l1_ratio=0.0, optimizer=optimizer.MomentumSGD(nesterovs=True))),
    (linear.LogisticRegression, dict(alpha=0.01, l1_ratio=0.0, optimizer=optimizer.Adam())),
    (linear.LogisticRegression, dict(alpha=0.01, l1_ratio=0.0, optimizer=optimizer.LBFGS())),
    (linear.LogisticRegression, dict(alpha=0.01, l1_ratio=0.5)),
    (linear.LogisticRegression, dict(alpha=0.01, l1_ratio=1.0)),
    (linear.LogisticRegression, dict(alpha=0.1, l1_ratio=0.0)),
    (linear.LogisticRegression, dict(alpha=0.1, l1_ratio=0.5)),
    (linear.LogisticRegression, dict(alpha=0.1, l1_ratio=1.0)),

    (svm.SVMClassifier, dict(C=1.0, kernel='linear', multi_class='ovr')),
    (svm.SVMClassifier, dict(C=1.0, kernel='linear', multi_class='ovo')),
    (svm.SVMClassifier, dict(C=1.0, kernel='polynomial', gamma=0.001, multi_class='ovr')),
    (svm.SVMClassifier, dict(C=1.0, kernel='polynomial', gamma=0.001, multi_class='ovo')),
    (svm.SVMClassifier, dict(C=1.0, kernel='sigmoid', gamma=0.001, multi_class='ovr')),
    (svm.SVMClassifier, dict(C=1.0, kernel='sigmoid', gamma=0.001, multi_class='ovo')),
    (svm.SVMClassifier, dict(C=1.0, kernel='rbf', gamma=0.001, multi_class='ovr')),
    (svm.SVMClassifier, dict(C=1.0, kernel='rbf', gamma=0.001, multi_class='ovo')),
    (svm.SVMClassifier, dict(C=10.0, kernel='rbf', gamma=0.001, multi_class='ovr')),
    (svm.SVMClassifier, dict(C=10.0, kernel='rbf', gamma=0.001, multi_class='ovo')),

    (decision_tree.DecisionTreeClassifier, dict(criterion='gini', splitter='best', min_impurity_decrease=0.0)),
    (decision_tree.DecisionTreeClassifier, dict(criterion='gini', splitter='best', min_impurity_decrease=1e-5)),
    (decision_tree.DecisionTreeClassifier, dict(criterion='entropy', splitter='best', min_impurity_decrease=0.0)),
    (decision_tree.DecisionTreeClassifier, dict(criterion='entropy', splitter='best', min_impurity_decrease=1e-5)),

    (decision_tree.DecisionTreeClassifier, dict(criterion='gini', splitter='random', max_features='log2')),
    (decision_tree.DecisionTreeClassifier, dict(criterion='gini', splitter='random', max_features='sqrt')),
    (decision_tree.DecisionTreeClassifier, dict(criterion='gini', splitter='random', max_features=0.2)),
    (decision_tree.DecisionTreeClassifier, dict(criterion='gini', splitter='random', max_features=30)),

    (adaboost.AdaBoostClassifier, dict(n_estimators=50, base_estimator=decision_tree.DecisionTreeClassifier(max_depth=1))),
    (adaboost.AdaBoostClassifier, dict(n_estimators=50, base_estimator=decision_tree.DecisionTreeClassifier(max_depth=2))),
    (adaboost.AdaBoostClassifier, dict(n_estimators=100, base_estimator=decision_tree.DecisionTreeClassifier(max_depth=1))),
    (adaboost.AdaBoostClassifier, dict(n_estimators=100, base_estimator=decision_tree.DecisionTreeClassifier(max_depth=2))),

    (gradient_boosting.GradientBoostingClassifier, dict(n_estimators=20, learning_rate=0.1)),
    (gradient_boosting.GradientBoostingClassifier, dict(n_estimators=20, learning_rate=0.2)),
    (gradient_boosting.GradientBoostingClassifier, dict(n_estimators=40, learning_rate=0.1)),
    (gradient_boosting.GradientBoostingClassifier, dict(n_estimators=40, learning_rate=0.2)),

    (random_forest.RandomForestClassifier, dict(n_estimators=10, max_features='log2')),
    (random_forest.RandomForestClassifier, dict(n_estimators=10, max_features='sqrt')),
    (random_forest.RandomForestClassifier, dict(n_estimators=20, max_features='log2')),
    (random_forest.RandomForestClassifier, dict(n_estimators=20, max_features='sqrt')),

    (neural_network.NeuralNetworkClassifier, dict(hidden_layer_sizes=(200,), alpha=0.001, optimizer=optimizer.SGD())),
    (neural_network.NeuralNetworkClassifier, dict(hidden_layer_sizes=(200,), alpha=0.001, optimizer=optimizer.MomentumSGD(nesterovs=False))),
    (neural_network.NeuralNetworkClassifier, dict(hidden_layer_sizes=(200,), alpha=0.001, optimizer=optimizer.MomentumSGD(nesterovs=True))),
    (neural_network.NeuralNetworkClassifier, dict(hidden_layer_sizes=(200,), alpha=0.001, optimizer=optimizer.Adam())),
    (neural_network.NeuralNetworkClassifier, dict(hidden_layer_sizes=(200,), alpha=0.001, optimizer=optimizer.LBFGS())),
    (neural_network.NeuralNetworkClassifier, dict(hidden_layer_sizes=(200,), alpha=0.01)),
    (neural_network.NeuralNetworkClassifier, dict(hidden_layer_sizes=(100, 100), alpha=0.001)),
    (neural_network.NeuralNetworkClassifier, dict(hidden_layer_sizes=(100, 100), alpha=0.01)),
]

print('start test')
l = []
for class_, params in class_params_list:
    l.append(test(class_, params, 800, 1000))
