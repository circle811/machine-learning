# Machine Learning in Python

## Introduction

- Pure python 3 code
- Only depend on python standard library and numpy
- Similar api to scikit-learn
- Effitent vectorized code
- Tested on real dataset

## Algorithms

|           | Optimizer                     |
|-----------|-------------------------------|
| optimizer | SGD, MomentumSGD, Adam, LBFGS |

|                   | Classification                         | Regression                |
|-------------------|----------------------------------------|---------------------------|
| knn               | KNNClassifier                          | KNNRegressor              |
| naive_bayes       | BernoulliNB, MultinomialNB, GaussianNB |                           |
| linear            | LogisticRegression                     | LinearRegression          |
| svm               | SVMClassifier                          |                           |
| decision_tree     | DecisionTreeClassifier                 | DecisionTreeRegressor     |
| adaboost          | AdaBoostClassifier                     |                           |
| gradient_boosting | GradientBoostingClassifier             | GradientBoostingRegressor |
| random_forest     | RandomForestClassifier                 | RandomForestRegressor     |
| neural_network    | NeuralNetworkClassifier                | NeuralNetworkRegressor    |

|                  | Clustering              |
|------------------|-------------------------|
| kmeans           | KMeans                  |
| gaussian_mixture | GaussianMixture         |
| agglomerative    | AgglomerativeClustering |
| dbscan           | DBSCAN                  |

|        | Dimensionality Reduction |
|--------|--------------------------|
| pca    | PCA, KernelPCA           |
| isomap | Isomap                   |
| tsne   | TSNE                     |

## Test

Requirements
- Python (3.6.5)
- numpy (1.14.3)
- matplotlib (2.2.2)

```shell
# clone source code
git clone https://github.com/circle811/machine-learning.git

cd machine-learning/
```

```shell
# download mnist dataset
mkdir -p datasets/mnist/

curl -o datasets/mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -o datasets/mnist/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -o datasets/mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -o datasets/mnist/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

```shell
# run test

# about 20 minutes
time python3 test/test_classification.py

# about 8 minutes
time python3 test/test_regression.py

# about 2 minutes
time python3 test/test_clustering.py

# about 2 minutes, need matplotlib
time python3 test/test_reduction.py
```
