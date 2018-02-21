"""Expose metrics not included by sklearn."""

import itertools

import numpy as np

from scipy.stats import ks_2samp

from sklearn.metrics import confusion_matrix

# pylama: ignore=D103


def ks_2samp_scorer(estimator, x, y, **predict_kwargs):
    """Return the 2 sample  Kolmogorov-Smirnov statistic test statistic.

    Examples:
        >>> import pandas as pd, numpy as np
        >>> from sklearn.tree import DecisionTreeClassifier as DTC
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import cross_val_score
        >>> dataset = load_breast_cancer()
        >>> clf = DTC()
        >>> score = cross_val_score(clf, dataset.data, dataset.target,
        ...                         scoring=ks_2samp_scorer)
        >>> np.mean(score) > 0.7
        True

    """
    assert len(estimator.classes_) == 2
    class0, class1 = estimator.classes_
    proba0 = estimator.predict_proba(x[y == class0], **predict_kwargs)[:, 1]
    proba1 = estimator.predict_proba(x[y == class1], **predict_kwargs)[:, 1]

    return ks_2samp(proba0, proba1)[0]  # return D only (discard p value).


def _get_probability_tresholds(y_score, max_points=100):
    unique_scores = np.sort(np.unique(np.concatenate(
        ([0, ],
         y_score,
         [1, ]))))
    # If the array is very big do a data reduction step.
    if len(unique_scores) > max_points:
        unique_scores = unique_scores[
            np.linspace(0, len(unique_scores) - 1, max_points, dtype=int)]
    return unique_scores


def _get_cm_based_score(y_true, y_score, function):
    """Get a confusion matrix based metric.

    Function should be applied to the decision matrix and return a float.

    """
    unique_scores = _get_probability_tresholds(y_score)
    cm_matrixes = map(
        confusion_matrix,
        itertools.repeat(y_true),
        map(lambda y_label, cutpoint: np.where(y_label <= cutpoint, 1, 0),
            itertools.repeat(y_score),
            unique_scores))
    # TODO: use a pandas dataframe with indexes to prevent confusion over
    # orientataion.
    # Make confusion matrixes true labels vertical.
    transposed_cm_matrixes = map(lambda x: x.T, cm_matrixes)
    return np.nan_to_num(  # Do not allow np.nan if there is zero division.
        np.fromiter(map(function, transposed_cm_matrixes), dtype=np.float))


def true_positive_rate(y_true, y_score):
    return _get_cm_based_score(
        y_true,
        y_score,
        lambda x: x[0][0] / x[:, 0].sum())


def false_positive_rate(y_true, y_score):
    return _get_cm_based_score(
        y_true,
        y_score,
        lambda x: x[0][1] / x[:, 1].sum())


def false_negative_rate(y_true, y_score):
    return 1 - true_positive_rate(y_true, y_score)


def true_negative_rate(y_true, y_score):
    return 1 - false_positive_rate(y_true, y_score)
