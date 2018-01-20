"""Optional file for XGBoost functionality."""
import functools
import copy

import numpy as np
import pandas as pd

from sklearn.metrics import get_scorer
# from sklearn.base import clone


def xgboost_get_feature_importances_from_booster(booster):
    """Get a feature importances dataframe from a booster object."""
    booster = copy.deepcopy(booster)
    score_types = ['weight',
                   'gain',
                   'cover']
    score_improved_labels = ['nr_occurences',
                             'average_gain',
                             'average_coverge']
    scores = map(lambda x: booster.get_score(importance_type=x),
                 score_types)
    # These are dictionaries with {feature1: value1, f2: v2, ...}.
    w, g, c = scores
    indexes = []
    rows = []
    for k in w.keys():
        indexes.append(k)
        rows.append((w[k], g[k], c[k]))
    df = pd.DataFrame.from_records(
        data=rows,
        index=indexes,
        columns=score_types)
    df.columns = score_improved_labels
    df['frequency'] = df['nr_occurences'] / df['nr_occurences'].sum()

    return df


def xgboost_get_learning_curve(estimator,
                               x_train,
                               x_test,
                               y_train,
                               y_test,
                               scoring='roc_auc'):
    """Return scores for test and training as as function of trees in XGBoost.

    Arguments:
        scoring_func (function or str): Same format as scikit-learn scorers.
        The signature of the function is ``(estimator, X, y)`` where
        ``estimator`` is the model to be evaluated, ``X`` is the test data and
        ``y`` is the ground truth labeling (or ``None`` in the case of
        unsupervised models).

    Returns:
        dict: keys: 'train_scores' and 'test_scores'. Values are the results of
        the scoring function.

    """
    # TODO: cloning the estimator removes its fitted parameters. On the one
    # hand this may prevent changing the estimator state on the other hand it
    # forces the estimator to be fit again (resource/consuming).
    # estimator = clone(estimator)
    predict_original = estimator.predict_proba
    scorer = get_scorer(scoring)
    ntrees = getattr(estimator, 'best_ntree_limit', estimator.n_estimators)
    test_list = list()
    train_list = list()
    for i in range(1, ntrees + 1):
        estimator.predict_proba = functools.partial(
            predict_original, ntree_limit=i)
        score_train = scorer(estimator, x_train, y_train)
        score_test = scorer(estimator, x_test, y_test)
        train_list.append(score_train)
        test_list.append(score_test)
    return dict(train_scores=np.array(train_list),
                test_scores=np.array(test_list))
