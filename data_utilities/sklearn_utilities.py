"""Scikit-learn utilities for common machine learning procedures."""
import itertools
import pickle
import os
import hashlib

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.model_selection import cross_val_score


def grid_search_cv(
        cv,
        estimator,
        param_grid,
        scoring,
        x_train,
        y_train,
        persistence_path=None):
    """Sklearn utilities version of grid search with cross validation."""
    all_parameters = param_grid.keys()
    all_values = param_grid.values()

    # Create a nice representation of estimator name.
    estimator_name = str(type(estimator))
    estimator_name = (
        estimator_name[
            estimator_name.index('\'')+1:
            estimator_name.index('\'',
                                 estimator_name.index('\'')+1)])
    estimator_name = estimator_name.replace('.', '_').lower()

    # Explore grid.
    grid_results = list()
    for i, one_grid_values in enumerate(itertools.product(*all_values)):
        one_grid_dict = dict(zip(all_parameters, one_grid_values))
        calculated_hash = _get_hash_from_dict(one_grid_dict)
        # Recover grids if they exist.
        if persistence_path is not None:
            grid_fname = os.path.join(
                persistence_path,
                '_'.join(('grid', estimator_name, str(i), '.pickle')))
            if os.path.isfile(grid_fname):
                with open(grid_fname, 'rb') as f:
                    loaded_dict = pickle.load(f)
                # Compute hash for this grid value.
                if loaded_dict['hash'] == calculated_hash:
                    grid_results.append(loaded_dict)
                    print(i)
                    continue
                else:
                    print('Calcualted hash is different than stored hash for {0}'.format(i))
        estimator.set_params(**one_grid_dict)
        scores = cross_val_score(estimator,
                                 x_train,
                                 y_train,
                                 scoring=scoring,
                                 cv=cv)
        one_grid_result = one_grid_dict
        one_grid_result.update({'scores': scores})
        one_grid_result.update({'hash': calculated_hash})
        # Save grid.
        if persistence_path is not None:
            with open(grid_fname, 'wb') as f:
                pickle.dump(one_grid_result, f, pickle.HIGHEST_PROTOCOL)
        grid_results.append(one_grid_result)

    # Order results.
    return sorted(grid_results, key=lambda x: np.mean(x['scores']))


def _get_hash_from_dict(dictionary):
    md5 = hashlib.md5()
    md5.update(pickle.dumps(dictionary))
    return md5.digest()


def execute_ks_2samp_over_time(time_series,
                               sample_label_series,
                               sample_series):
    """Execute a Kolmogorov-Smirnov 2 sample test on a time series."""
    # Last provided series is usually a result of a clf.predict_proba and it
    # has a different index than the other two series. Correct this.
    sample_series.index = time_series.index

    l1, l2 = sorted(sample_label_series.unique())
    time_ks_d_value = list()  # a list of (time, D) values.
    for time in time_series.sort_values().unique():
        # Subset entire input series into time slices.
        sub_sample_series = sample_series.loc[time_series == time]
        # Further divide our samples based on labels.
        s1 = sub_sample_series.loc[sample_label_series == l1]
        s2 = sub_sample_series.loc[sample_label_series == l2]
        if s1.empty or s2.empty:
            time_ks_d_value.append((time, np.nan))
        else:
            D, p_value = scipy.stats.ks_2samp(s1, s2)
            time_ks_d_value.append((time, D))
    unzipped = tuple(zip(*time_ks_d_value))
    return pd.Series(data=unzipped[1], index=unzipped[0])


def get_sorted_feature_importances(classifier, attributes):
    """Return a sorted list of feature importances."""
    v = classifier.feature_importances_
    k = attributes
    assert(len(v) == len(k))
    unordered = tuple(zip(k, v))
    ordered = sorted(unordered, key=lambda x: x[1], reverse=False)
    return ordered


def xgboost_get_feature_importances_from_booster(booster):
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
    df['frequency'] = df['weight'] / df['weight'].sum()

    return df
