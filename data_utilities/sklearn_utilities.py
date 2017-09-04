"""Scikit-learn utilities for common machine learning procedures."""
import itertools
import pickle
import os

import numpy as np

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

    # Explore grid.
    grid_results = list()
    for i, one_grid_values in enumerate(itertools.product(*all_values)):
        if persistence_path is not None:
            grid_fname = os.path.join(persistence_path,
                                      'grid_' + str(i) + '.pickle')
            if os.path.isfile(grid_fname):
                with open(grid_fname, 'rb') as f:
                    grid_results.append(pickle.load(f))
                print(i)
                continue
        one_grid_dict = dict(zip(all_parameters, one_grid_values))
        estimator.set_params(**one_grid_dict)
        scores = cross_val_score(estimator,
                                 x_train,
                                 y_train,
                                 scoring=scoring,
                                 cv=cv)
        one_grid_result = one_grid_dict
        one_grid_result.update({'scores': scores})
        if persistence_path is not None:
            with open(grid_fname, 'wb') as f:
                pickle.dump(one_grid_result, f, pickle.HIGHEST_PROTOCOL)
        grid_results.append(one_grid_result)

    # Order results.
    return sorted(grid_results, key=lambda x: np.mean(x['scores']))
