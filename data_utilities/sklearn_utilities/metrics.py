"""Expose metrics not included by sklearn."""

from scipy.stats import ks_2samp


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
