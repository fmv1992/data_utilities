"""Offer an evolutionary grid search object estimator.

This script follows the guidelines of the sklearn project:
http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects

The idea is that the EvolutionaryPersistentGridSearchCV object contains a an
attribute that is the persistent evolutionary object.

Its fit method would call an evolutionary function that would store
intermediate results in the persistent evolutionary object.

>>> import numpy as np
>>> from sklearn.base import BaseEstimator, ClassifierMixin
>>> from sklearn.utils.validation import check_X_y, check_array, check_is_fitted  # noqa
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

import random
import os
import multiprocessing as mp
import hashlib

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.grid_search import _check_param_grid


from deap.algorithms import eaSimple

from data_utilities.sklearn_utilities.grid_search import BasePersistentGrid

# pylama: ignore=D103,D102,W0611


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
                 ef_kwargs=dict(),
                 persistent_grid_path=None,
                 dataset_path=None,
                 hash_function=hashlib.md5,
                 save_every_n_interations=10):
        # Call super init.
        super().__init__(
            persistent_grid_path=persistent_grid_path,
            dataset_path=dataset_path,
            hash_function=hash_function,
            save_every_n_interations=save_every_n_interations)
        # Store arguments.
        self.ev_func = ev_func
        self.ef_args = ef_args
        self.ef_kwargs = ef_kwargs
        # TODO: precise and flexible (see below).
        self.population = ef_args[0]   # TODO: be more precise when getting pop
        self.toolbox = ef_args[1]   # TODO: be more precise when getting toolbo
        self.cxpb = ef_args[2]   # TODO: see above
        self.mutpb = ef_args[3]   # TODO: see above
        self.ngen = ef_args[4]   # TODO: see above
        # Store correct value.
        self._ngen = self.ngen
        # Create a dictionary of the form:
        # {'argname': passed_value, ..., 'argname': default_if_non_passed}
        # Base hash:
        # The base hash is the hash that is combined with any new
        # parameters go guarantee that the combination of
        # hash(base_hash + hash(parameter)) is unique.
        # For the case of the EvolutionaryPersistentGrid the base hash is the
        # hash(hash(dataset) + hash(arg1) + hash(arg2) + ...
        # + hash(function name).
        self.hash_sequence = (dataset_path,
                              ef_args,
                              ef_kwargs,
                              ev_func.__name__,)
        self.base_hash = self.get_hash(b''.join(
            map(self.get_hash, map(self._transform_to_hashable,
                                   self.hash_sequence))))

    def _paraellize_toolbox(self):
        if os.name != 'nt':  # Use multiprocessing if not on Windows.
            self.toolbox.unregister('map')
            self.pool = mp.Pool()
            # TODO: allow customization of pool
            self.toolbox.register('map', self.pool.map)
        else:
            self.pool = None

    def _update_base_hash(self, x):
        self.base_hash = self.get_hash(b''.join(
            map(self.get_hash, map(self._transform_to_hashable,
                                   self.hash_sequence))))


class EvolutionaryPersistentGridSearchCV(GridSearchCV):
    """Perform an evolutionary grid search.

    Also perform with cross validation and persistence.

    """

    def __init__(self,
                 persistent_evolutionary_object,
                 estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
        super().__init__(
            estimator, scoring, fit_params, n_jobs, iid,
            refit, cv, verbose, pre_dispatch, error_score)
        self.epgo = persistent_evolutionary_object
        _check_param_grid(param_grid)

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Store the data.
        self.X_ = X
        self.y_ = y

        # Iterate over populations.
        return self._fit(self)

        # Return the classifier
        return self

    def _fit(self):
        pass

    def _evolve(self):
        # Initialize population.
        pass
        # Execute evolution cycles.
        evolution_full_cycles = (
            self.epgo.ngen // self.epgo.save_every_n_interations)
        remaining_cycles = (
            self.epgo.ngen % self.epgo.save_every_n_interations)
        for _ in range(evolution_full_cycles):
            ef_args = (self.ef_args[0],
                       self.ef_args[1],
                       self.ef_args[2],
                       self.ef_args[3],
                       self.epgo.save_every_n_interations)
            self.epgo.pop = self.epgo.ev_func(ef_args, **self.epgo.ef_kwargs)
            self.epgo.save()

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


class IndividualFromGrid(object):
    """Individual object to be created from grid."""

    def __init__(self,
                 grid):
        """Create an individuals to evolve from a grid dictionary.

        Arguments:
            grid (dict): dictionary mapping hyperparameters to either a two
            element tuple (min, max) for continuous variables or a set of
            values.

        Returns:
            object: object with attributes whose names are the keys from the
            dictionary and either uniformly drawn numbers between (min, max) or
            elements from given sets.

        """
        self.data = self._init_from_grid(grid)

    def _init_from_grid(self, grid):
        return {k: self._switch_function(v) for k, v in grid.items()}

    @classmethod
    def _switch_function(cls, value):
        if isinstance(value, tuple):
            return cls._init_from_tuple(value)
        elif isinstance(value, set) or isinstance(value, frozenset):
            return cls._init_from_set(value)
        else:
            raise NotImplementedError

    @classmethod
    def _init_from_tuple(cls, arg_tup):
        return random.uniform(*arg_tup)

    @classmethod
    def _init_from_set(cls, arg_set):
        return random.sample(arg_set, 1)[0]
