"""Offer an evolutionary grid search object estimator.

This script follows the guidelines of the sklearn project:
http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects

>>> import numpy as np
>>> from sklearn.base import BaseEstimator, ClassifierMixin
>>> from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
>>> from sklearn.utils.multiclass import unique_labels
>>> from sklearn.metrics import euclidean_distances
>>> class TemplateClassifier(BaseEstimator, ClassifierMixin):
...
...     def __init__(self, demo_param='demo'):
...         self.demo_param = demo_param
...
...     def fit(self, X, y):
...
...         # Check that X and y have correct shape
...         X, y = check_X_y(X, y)
...         # Store the classes seen during fit
...         self.classes_ = unique_labels(y)
...
...         self.X_ = X
...         self.y_ = y
...         # Return the classifier
...         return self
...
...     def predict(self, X):
...
...         # Check is fit had been called
...         check_is_fitted(self, ['X_', 'y_'])
...
...         # Input validation
...         X = check_array(X)
...
...         closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
...         return self.y_[closest]

"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

import hashlib

from deap.algorithms import eaSimple

from data_utilities.sklearn_utilities.grid_search import BasePersistentGrid

def ea_simple_worker():
    pass

def ea_simple_with_persistence(evolutionary_persistent_object,
                               # Be transparent, put all args for function
                               # here.
                               n_jobs=-1,
                               #
                               stats=None,
                               halloffame=None,
                               verbose=__debug__):
    """Reproduce eaSimple from deap with persistence."""
    pass

def _func_args_to_dict(function, *func_args, **func_kwargs):
    varnames = function.__code__.co_varnames
    defaults = function.__defaults__
    actual_call_values = func_args + defaults[len(func_args):]
    return dict(zip(varnames, actual_call_values))

class EvolutionaryPersistentGrid(BasePersistentGrid):
    """Store all data necessary to restore an evolution call.

    This object assumes that both the data and the evolutionary parameters can
    change. Thus it compartimentalizes the combination of (data + function +
    parameters) when loading stored populations.

    Store:
        * population
        * generation number
    (to check whether continuation is allowed)
        * cross over probability
        * mutation probability
        * toolbox
    (optional)
        * best individual

    """
    def __init__(self,
                 ev_func,
                 ef_args=tuple(),
                 persistent_grid_path=None,
                 dataset_path=None,
                 hash_function=hashlib.md5,
                 save_every_n_interations=10):
        super(EvolutionaryPersistentGrid, self).__init__(
            persistent_grid_path=persistent_grid_path,
            dataset_path=dataset_path,
            hash_function=hash_function,
            save_every_n_interations=save_every_n_interations)
        # Create a dictionary of the form:
        # {'argname': passed_value, ..., 'argname': default_if_non_passed}
        # Base hash:
        # The base hash is the hash that is combined with any new
        # parameters go guarantee that the combination of
        # hash(base_hash + hash(parameter)) is unique.
        # For the case of the EvolutionaryPersistentGrid the base hash is the
        # hash(hash(dataset) + hash(arg1) + hash(arg2) + ...
        # + hash(function name).
        hash_sequence = (dataset_path, ) + ef_args + (ev_func.__name__, )
        self.base_hash = self.get_hash(b''.join(
            map(self.get_hash, map(self._transform_to_hashable,
                                   hash_sequence))))

    def update(self, estimator, grid, results, population):
        # call super update without population.
        # store population
        pass

    def save(self):
        # store population as builtins list
        # call super save
        pass

class EvolutionaryPersistentGridSearchCV(GridSearchCV):
    """Perform an evolutionary grid search.

    Also perform with cross validation and persistence.

    """
    def __init__(self,
                 persistent_evolutionary_object,
                 estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
        super(EvolutionaryPersistentGridSearchCV, self).__init__(
            estimator, scoring, fit_params, n_jobs, iid,
            refit, cv, verbose, pre_dispatch, error_score)
        self.persistent_evolutionary_object = persistent_evolutionary_object
        _check_param_grid(param_grid)

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


    def _parse_grid_dtypes(self):
        pass
