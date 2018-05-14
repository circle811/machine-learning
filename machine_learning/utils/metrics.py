import numpy as np

from .distance import l2_distance, pairwise_distance_function

__all__ = ['confusion_matrix',
           'accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'fbeta_score', 'log_loss',
           'explained_variance_score', 'r2_score', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
           'contingency_matrix',
           'rand_score', 'adjusted_rand_score', 'fowlkes_mallows_score',
           'mutual_info_score', 'normalized_mutual_info_score',
           'homogeneity_score', 'completeness_score', 'v_measure_score', 'homogeneity_completeness_v_measure',
           'silhouette_samples', 'silhouette_score', 'calinski_harabaz_score']

TINY = np.finfo(np.float64).tiny


def confusion_matrix(y_true, y_pred):
    n_samples = y_true.shape[0]
    classes, yi = np.unique(np.concatenate([y_true, y_pred]), return_inverse=True)
    n_classes = classes.shape[0]
    yi_true = yi[:n_samples]
    yi_pred = yi[n_samples:]
    return np.bincount(yi_true * n_classes + yi_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='binary'):
    c = confusion_matrix(y_true, y_pred)
    p = c.diagonal() / np.maximum(1, c.sum(axis=0))
    if average is None:
        return p
    elif average == 'binary':
        return p[1]
    elif average == 'micro':
        return np.sum(c.diagonal()) / y_true.shape[0]
    elif average == 'macro':
        return np.mean(p)
    else:
        raise ValueError('average')


def recall_score(y_true, y_pred, average='binary'):
    c = confusion_matrix(y_true, y_pred)
    r = c.diagonal() / np.maximum(1, c.sum(axis=1))
    if average is None:
        return r
    elif average == 'binary':
        return r[1]
    elif average == 'micro':
        return np.sum(c.diagonal()) / y_true.shape[0]
    elif average == 'macro':
        return np.mean(r)
    else:
        raise ValueError('average')


def f1_score(y_true, y_pred, average='binary'):
    c = confusion_matrix(y_true, y_pred)
    p = c.diagonal() / np.maximum(1, c.sum(axis=0))
    r = c.diagonal() / np.maximum(1, c.sum(axis=1))
    f = 2 * p * r / np.maximum(TINY, p + r)
    if average is None:
        return f
    elif average == 'binary':
        return f[1]
    elif average == 'micro':
        return np.sum(c.diagonal()) / y_true.shape[0]
    elif average == 'macro':
        return np.mean(f)
    else:
        raise ValueError('average')


def fbeta_score(y_true, y_pred, beta, average='binary'):
    c = confusion_matrix(y_true, y_pred)
    p = c.diagonal() / np.maximum(1, c.sum(axis=0))
    r = c.diagonal() / np.maximum(1, c.sum(axis=1))
    f = (1 + beta ** 2) * p * r / np.maximum(TINY, beta ** 2 * p + r)
    if average is None:
        return f
    elif average == 'binary':
        return f[1]
    elif average == 'micro':
        return np.sum(c.diagonal()) / y_true.shape[0]
    elif average == 'macro':
        return np.mean(f)
    else:
        raise ValueError('average')


def log_loss(y_true, y_pred, eps=1e-15):
    n_samples = y_true.shape[0]
    _, yi = np.unique(y_true, return_inverse=True)
    p = y_pred[np.arange(n_samples), yi]
    return -np.mean(np.log(np.clip(p, eps, 1 - eps)))


# regression
def explained_variance_score(y_true, y_pred):
    return 1 - np.var(y_true - y_pred) / np.var(y_true)


def r2_score(y_true, y_pred):
    return 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def median_absolute_error(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))


# clustering
def contingency_matrix(labels_true, labels_pred):
    classes_true, i_true = np.unique(labels_true, return_inverse=True)
    classes_pred, i_pred = np.unique(labels_pred, return_inverse=True)
    n_true = classes_true.shape[0]
    n_pred = classes_pred.shape[0]
    return np.bincount(i_true * n_pred + i_pred, minlength=n_true * n_pred).reshape(n_true, n_pred)


def pairwise_count(labels_true, labels_pred):
    n_samples = labels_true.shape[0]
    n_pairs = n_samples * (n_samples - 1) // 2
    c = contingency_matrix(labels_true, labels_pred)
    ss = np.sum(c * (c - 1)) // 2
    sd = np.sum(c * (np.sum(c, axis=1)[:, np.newaxis] - c)) // 2
    ds = np.sum(c * (np.sum(c, axis=0) - c)) // 2
    dd = n_pairs - ss - sd - ds
    return ss, sd, ds, dd


def entropy(c):
    p = c / np.maximum(TINY, np.sum(c))
    lp = np.log(p, out=np.zeros_like(p), where=p > 0)
    return -np.sum(p * lp)


def entropy_and_mutual_info(labels_true, labels_pred):
    c = contingency_matrix(labels_true, labels_pred)
    a = np.sum(c, axis=0)
    b = np.sum(c, axis=1)
    ha = entropy(a)
    hb = entropy(b)
    mi = ha + hb - entropy(c)
    return ha, hb, mi


def rand_score(labels_true, labels_pred):
    ss, sd, ds, dd = pairwise_count(labels_true, labels_pred)
    return (ss + dd) / (ss + sd + ds + dd)


def adjusted_rand_score(labels_true, labels_pred):
    n_samples = labels_true.shape[0]
    n_pairs = n_samples * (n_samples - 1) // 2
    c = contingency_matrix(labels_true, labels_pred)
    a = np.sum(c, axis=0)
    b = np.sum(c, axis=1)
    comb_c = np.sum(c * (c - 1)) // 2
    comb_a = np.sum(a * (a - 1)) // 2
    comb_b = np.sum(b * (b - 1)) // 2
    prod = comb_a * comb_b / n_pairs
    mean = (comb_a + comb_b) / 2
    return (comb_c - prod) / (mean - prod)


def fowlkes_mallows_score(labels_true, labels_pred):
    ss, sd, ds, dd = pairwise_count(labels_true, labels_pred)
    if ss == 0:
        return 0.0
    else:
        return ss / np.sqrt((ss + sd) * (ss + ds))


def mutual_info_score(labels_true, labels_pred):
    ha, hb, mi = entropy_and_mutual_info(labels_true, labels_pred)
    return mi


def normalized_mutual_info_score(labels_true, labels_pred):
    ha, hb, mi = entropy_and_mutual_info(labels_true, labels_pred)
    return mi / np.sqrt(ha * hb)


def homogeneity_score(labels_true, labels_pred):
    ha, hb, mi = entropy_and_mutual_info(labels_true, labels_pred)
    h = mi / hb
    return h


def completeness_score(labels_true, labels_pred):
    ha, hb, mi = entropy_and_mutual_info(labels_true, labels_pred)
    c = mi / ha
    return c


def v_measure_score(labels_true, labels_pred):
    ha, hb, mi = entropy_and_mutual_info(labels_true, labels_pred)
    h = mi / hb
    c = mi / ha
    v = 2 * h * c / (h + c)
    return v


def homogeneity_completeness_v_measure(labels_true, labels_pred):
    ha, hb, mi = entropy_and_mutual_info(labels_true, labels_pred)
    h = mi / hb
    c = mi / ha
    v = 2 * h * c / (h + c)
    return h, c, v


def silhouette_samples(X, labels, metric='l2'):
    n_samples = X.shape[0]
    a = np.arange(n_samples)
    d = pairwise_distance_function[metric](X, X)
    labels_u, labels_i, labels_c = np.unique(labels, return_inverse=True, return_counts=True)
    n_labels = labels_u.shape[0]

    s = np.zeros(n_samples)
    for i in range(n_samples):
        li = labels_i[i]
        dc = np.zeros((n_samples, n_labels))
        dc[a, labels_i] = d[i]
        dc_sum = np.sum(dc, axis=0)
        dc_self = dc_sum[li] / max(labels_c[li] - 1, 1)
        dc_mean = dc_sum / labels_c
        dc_mean[li] = np.inf
        dc_min = np.min(dc_mean)
        s[i] = (dc_min - dc_self) / max(dc_self, dc_min)

    return s


def silhouette_score(X, labels, metric='l2'):
    return np.mean(silhouette_samples(X, labels, metric))


def calinski_harabaz_score(X, labels):
    n_samples = X.shape[0]
    labels_u, labels_i, labels_c = np.unique(labels, return_inverse=True, return_counts=True)
    n_labels = labels_u.shape[0]
    center = np.mean(X, axis=0)
    w = 0.0
    b = 0.0
    for k in range(n_labels):
        Xk = X[labels_i == k]
        center_k = np.mean(Xk, axis=0)
        w += np.sum(l2_distance(Xk, center_k, square=True))
        b += labels_c[k] * l2_distance(center_k, center, square=True)
    return 1.0 if w == 0 else b * (n_samples - n_labels) / (w * (n_labels - 1))
