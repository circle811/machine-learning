import numpy as np

__all__ = ['explained_variance_score', 'r2_score', 'mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
           'rand_score', 'adjusted_rand_score', 'fowlkes_mallows_score',
           'mutual_info_score', 'normalized_mutual_info_score',
           'homogeneity_score', 'completeness_score', 'v_measure_score', 'homogeneity_completeness_v_measure']

TINY = np.finfo(np.float64).tiny


# classification


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
