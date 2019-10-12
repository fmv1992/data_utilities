"""Scikit-learn utilities for common machine learning procedures."""
import hashlib
import itertools
import multiprocessing
import os
import pickle
import threading
import warnings
from importlib.util import find_spec  # Check for presence of optional deps.

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.model_selection import cross_val_score

from data_utilities import python_utilities as pyu

# Imported at the bottom of the file.
# from . import grid_search
# from . import evolutionary_grid_search
# from . import _evolutionary_grid_search as evolutionary_grid_search
# from ._xgboost import (xgbfunctions here)


def multiprocessing_grid_search(queue, shared_list, persistent_object):
    """Explore cross validation grid using multiprocessing."""
    # scores = cross_val_score(*cross_val_score_args, **cross_val_score_kwargs)
    # queue.put(scores)
    while True:
        passed_parameters = queue.get()
        if passed_parameters is None:
            break
        # Dismember arguments and values.
        grid, cvs_args, cvs_kwargs = passed_parameters
        estimator, x = cvs_args
        estimator.set_params(**grid)
        del cvs_args

        # Check if value was already calculated:
        stored_value = persistent_object.retrieve(estimator, grid)
        if stored_value is None:
            scores = cross_val_score(estimator, x, **cvs_kwargs)
            persistent_object.update(estimator, grid, scores)
        else:
            scores = stored_value
        grid_result = grid.copy()
        grid_result['scores'] = scores
        shared_list.append(grid_result)


# TODO: Do one thing and to it well.
#       Yet try to add some flexibility in using the 'cross_validate' function
#       to be able to synthesize artificial metrics (such as a combination of
#       ROC AUC and KS-2-sample.
#       Maybe add an artificial sorting function instead of the numpy.mean.
#       For better API maybe change *args, **kwargs to their own arguments.
#       This enables the cross_val_function to be a kwargs as well as the
#       sorting function.
#
#       Add verbosity parameter to this function.
def persistent_grid_search_cv(persistent_object,
                              grid_space,
                              *cross_val_score_args,
                              **cross_val_score_kwargs):
    """Sklearn utilities version of grid search with cross validation.

    Sklearns' cross_val_score args and kwargs:
        * estimator
        * X
        * y=None
        * scoring=None
        * cv=None
        * n_jobs=1
        * verbose=0
        * fit_params=None
        * pre_dispatch='2*n_jobs'

    """
    # Dismember arguments and values.
    n_workers = _get_n_jobs_as_joblib_parallel(
        cross_val_score_kwargs.get('n_jobs', -1))
    # This function already creates 4 parallel works. In order to avoid having
    # n**2 parallel workers then reset n_jobs.
    cross_val_score_kwargs['n_jobs'] = 1

    # Dismember grid space.
    all_parameters = grid_space.keys()
    all_values = grid_space.values()

    # Initialize multiprocessing manager, queue, shared list, and enable
    # multiprocessing capabilities on persistent_object.
    mp_manager = persistent_object.get_multiprocessing_manager()
    # Initialize queue.
    mp_queue = mp_manager.Queue(2 * n_workers)
    # Initialize shared list.
    mp_scores_list = mp_manager.list()
    # Enable multiprocessing capabilities on persistent_object.
    # Start parallel workers.
    jobs = []
    if os.name == 'nt':  # if on windows use threading. Jesus, Windows...
        warnings.warn('Mocking multiprocessing capabilities on Windows using '
                      'threading.',
                      UserWarning)
        p = threading.Thread(
            target=multiprocessing_grid_search,
            args=(mp_queue, mp_scores_list, persistent_object),
            kwargs={})
        p.start()
        jobs.append(p)
    else:
        for i in range(n_workers):
            p = multiprocessing.Process(
                target=multiprocessing_grid_search,
                args=(mp_queue, mp_scores_list, persistent_object),
                kwargs={})
            p.start()
            jobs.append(p)
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
    # Save persistent grid object.
    persistent_object.save()
    # Order results.
    return sorted(list(mp_scores_list), key=lambda x: np.mean(x['scores']))


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


def get_feature_importances_dataframe(classifier, features):
    """Return dataframe with features and their importances."""
    df = pd.DataFrame(
        data={'feature_importances': classifier.feature_importances_},
        index=features)
    df.index.name = 'feature'
    return df


def get_estimator_name(estimator):
    """Get a simple representation of an estimator's name."""
    estimator_name = pyu.process_string(estimator.__class__.__name__)
    return estimator_name


def _get_n_jobs_as_joblib_parallel(n):
    """Return the number of jobs according to joblib.Parallel rules."""
    if n == -1:
        n_workers = multiprocessing.cpu_count()
    elif n < 0:
        n_workers = (multiprocessing.cpu_count() + 1 + n)
    elif n > 0:
        n_workers = n
    return n_workers


def get_confusion_matrix(*args, **kwargs):
    """Get a confusion matrix with predefined names.

    Remove ambiguity from confusion matrix by using fixed names for actual
    labels and predicted labels.

    The predicted axis will be set to 'predicted'.
    The ground truth axis will be set to 'actual'.

    The kwargs 'rownames' and 'colnames' are ignored.

    """

    # In case shapes are just 'unaligned', e.g.: (300,1) versus (1,300).
    if (args[0].shape != args[1].shape) and (
            np.prod(args[0].shape) == np.prod(args[1].shape)):
        args = (np.ravel(args[0]), np.ravel(args[1])) + args[2:]

    index = args[0]
    columns = args[1]

    kwargs['rownames'] = ['actual', ]
    kwargs['colnames'] = ['predicted', ]

    raw_tab = pd.crosstab(*args, **kwargs)

    return raw_tab


if __name__ != '__main__':
    # Running this file will cause import errors.
    from . import grid_search  # noqa
    from . import plot  # noqa
    from . import metrics  # noqa

    # Importing of optional dependencies.
    if find_spec('deap') is not None:
        from . import _evolutionary_grid_search as evolutionary_grid_search  # noqa
    if find_spec('xgboost') is not None:
        from ._xgboost import (xgboost_get_feature_importances_from_booster,  # noqa
                               xgboost_get_learning_curve)
