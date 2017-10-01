"""Scikit-learn utilities for common machine learning procedures."""
import hashlib
import itertools
import pickle
import multiprocessing

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.model_selection import cross_val_score

from data_utilities import python_utilities as pyu

from data_utilities.sklearn_utilities import grid_search


def multiprocessing_grid_search(queue, shared_list, persistence_object):
    """Explore cross validation grid using multiprocessing."""
    # scores = cross_val_score(*cross_val_score_args, **cross_val_score_kwargs)
    # queue.put(scores)
    while True:
        # All parameters from cross_val_score, i to compute pickle name and
        # persistence_path.
        passed_parameters = queue.get()
        if passed_parameters is None:
            break
        # Dismember arguments and values.
        grid, cvs_args, cvs_kwargs = passed_parameters
        estimator, x = cvs_args
        estimator.set_params(**grid)
        del cvs_args

        # Check if value was already calculated:
        stored_value = persistence_object.retrieve(estimator, grid)
        if stored_value is None:
            scores = cross_val_score(estimator, x, **cvs_kwargs)
            persistence_object.update(estimator, grid, scores)
        else:
            scores = stored_value
        grid_result = grid.copy()
        grid_result['scores'] = scores
        shared_list.append(grid_result)


def persistent_grid_search_cv(persistence_object,
                   grid_space,
                   *cross_val_score_args,
                   **cross_val_score_kwargs):
    """Sklearn utilities version of grid search with cross validation."""
    # Dismember arguments and values.
    if 'n_jobs' in cross_val_score_kwargs.keys():
        if cross_val_score_kwargs['n_jobs'] == -1:
            n_workers = multiprocessing.cpu_count()
        elif cross_val_score_kwargs['n_jobs'] < 0:
            n_workers = multiprocessing.cpu_count() + 1 + cross_val_score_kwargs['n_jobs']
        elif cross_val_score_kwargs['n_jobs'] > 0:
            n_workers = cross_val_score_kwargs['n_jobs']
    else:
        n_workers = multiprocessing.cpu_count()
    all_parameters = grid_space.keys()
    all_values = grid_space.values()

    # Initialize multiprocessing manager, queue, shared list, and enable
    # multiprocessing capabilities on persistence_object.
    mp_manager = persistence_object.get_multiprocessing_manager()
    # Initialize queue.
    mp_queue = mp_manager.Queue(64)
    # Initialize shared list.
    mp_scores_list = mp_manager.list()
    # Enable multiprocessing capabilities on persistence_object.
    # Start parallel workers.
    jobs = []
    for i in range(n_workers):
        p = multiprocessing.Process(
            target=multiprocessing_grid_search,
            args=(mp_queue, mp_scores_list, persistence_object),
            kwargs={})
        p.start()
        jobs.append(p)
    # Explore grid.
    grid_results = list()
    # Iterate over grid values.
    for i, one_grid_values in enumerate(itertools.product(*all_values)):
        # Create a dict from values.
        one_grid_dict = dict(zip(all_parameters, one_grid_values))
        # Feed it into the queue.
        mp_queue.put((one_grid_dict,
                      cross_val_score_args,
                      cross_val_score_kwargs))
    # Close opened processes.
    for p in jobs:
        mp_queue.put(None)
    for p in jobs:
        p.join()
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


def get_estimator_name(estimator):
    """Get a good representation of estimator name."""
    # Create a nice representation of estimator name.
    estimator_name = pyu.process_string(str(estimator)).strip('_')
    index_last_underscore = (len(estimator_name)
                             - estimator_name[::-1].index('_'))
    estimator_name = estimator_name[index_last_underscore:]
    return estimator_name


def xgboost_get_feature_importances_from_booster(booster):
    """Get a feature importances dataframe from a booster object."""
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
    df['frequency'] = df['weight'] / df['weight'].sum()

    return df
