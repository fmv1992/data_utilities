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
import itertools

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.grid_search import _check_param_grid
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin


from deap.algorithms import eaSimple
import deap.base
import deap.creator
import deap.tools

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
        self.toolbox = ef_args[1]
        self.pop = self.toolbox.pop
        self.cxpb = ef_args[2]
        self.mutpb = ef_args[3]
        self.ngen = ef_args[4]
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


class EvolutionaryPersistentGridSearchCV(BaseEstimator, ClassifierMixin):
    """Perform an evolutionary grid search.

    Also perform with cross validation and persistence.

    """

    def __init__(self,
                 persistent_evolutionary_object,
                 estimator, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):

        super().__init__()

        self.epgo = persistent_evolutionary_object
        self.epgo.toolbox.estimator = estimator
        self.scoring = scoring

    def fit(self, x, y):

        # Check that x and y have correct shape
        x, y = check_X_y(x, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Store the data.
        self.x_ = x
        self.y_ = y

        # Iterate over populations.
        self._fit()

        # Return the classifier
        return self

    def _fit(self):
        self._evolve()

    def _evolve(self):
        # Make individuals point to x and y.
        for ind in self.epgo.pop:
            ind.x = self.x_
            ind.y = self.y_
        # Execute evolution cycles.
        evolution_full_cycles = (
            self.epgo.ngen // self.epgo.save_every_n_interations)
        remaining_cycles = (
            self.epgo.ngen % self.epgo.save_every_n_interations)
        for _ in range(evolution_full_cycles):
            self.epgo.pop = eaSimple(
                self.epgo.pop,
                self.epgo.toolbox,
                self.epgo.cxpb,
                self.epgo.mutpb,
                self.epgo.ngen)
            # TODO: remove references of x and y from individuals.
            self.epgo.save()

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


class EvolutionaryToolbox(deap.base.Toolbox):
    """Evolutionary toolbox with sklearn_utilities defaults."""

    def __init__(self,
                 grid,
                 combiner=None,
                 mutator=None,
                 population=None,
                 estimator=None,
                 ):

        super().__init__()

        self.estimator = estimator

        self.register('mutate', mutator.mutate)
        self.register('combine', combiner.combine)
        self.mate = self.combine
        self.register('select', deap.tools.selTournament, tournsize=3)
        if isinstance(population, int):
            self.pop = self._create_population(grid, population)
        else:
            self.pop = population
        # TODO: cover both cases of None and not None.

    def _create_population(self, grid, n_individuals):
        deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
        return list(map(IndividualFromGrid, itertools.repeat(grid,
                                                             n_individuals)))

    def reset_grid_values(self, individual):
        individual.estimator.set_params(**individual.data)

    def evaluate(self,
                 individual,
                 *cross_val_score_args,
                 **cross_val_score_kwargs):
        self.estimator.set_params(**individual.data)
        self.estimator.fit(individual.x, individual.y)
        return (np.mean(
            cross_val_score(self.estimator, individual.x, individual.y,
                            *cross_val_score_args, **cross_val_score_kwargs)),
                )


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
        self.fitness = deap.creator.FitnessMin()
        self.fitness.weights = (-1.0, )

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


class EvolutionaryMutator(object):
    """Mutator object that mutates individuals from grid.

    Arguments:
        grid (dict): dictionary mapping hyperparameters to either a two
        element tuple (min, max) for continuous variables or a set of
        values.
        kwargs (dict): dictionary mapping data types to functions that mutate
        individuals.
    """

    def __init__(self,
                 grid,
                 kwargs=dict()):
        self.grid = grid
        self._key_to_dtypes = self._get_dtypes(grid)
        self._dtypes_to_func = {
            int: self._mutate_int,
            float: self._mutate_float,
            str: self._mutate_str,
            tuple: self._mutate_tuple,
            list: self._mutate_list,
            set: self._mutate_set, }
        self._dtypes_to_func.update(kwargs)

    def mutate(self, individual):
        # TODO: Maybe do some performance increase here... However mutation is
        # not so frequent.
        for key in individual.data.keys():
            mutation_func = self._dtypes_to_func[self._key_to_dtypes[key]]
            # If is modifiable.
            mutated_value = mutation_func(individual.data[key])
            if mutated_value is not None:  # That means object is not mutable.
                individual.data[key] = mutated_value
        return (individual, )

    def _mutate_int(self, value):
        increment = int(np.random.normal(loc=0, scale=2))
        while increment == 0:
            increment = int(np.random.normal(loc=0, scale=2))
        return  value + increment

    def _mutate_float(self, value):
        increment = np.random.normal(loc=0, scale=2)
        while increment == 0:
            increment = np.random.normal(loc=0, scale=2)
        return value + increment

    def _mutate_str(self, value):
        raise NotImplementedError
        return None

    def _mutate_tuple(self, value):
        raise NotImplementedError
        return None

    def _mutate_list(self, value):
        raise NotImplementedError
        return None

    def _mutate_set(self, value):
        raise NotImplementedError
        return None

    def _get_dtypes(self, grid):
        dtypes_dict = dict()
        for key, value in grid.items():
            for dtype in (int, float, str, tuple, list, set, frozenset):
                if isinstance(random.sample(value, 1)[0], dtype):
                    dtypes_dict[key] = dtype
                    break
            else:  # exhausted loop and found no dtype.
                raise NotImplementedError(
                    'No dtype for {0} was found.'.format(value))
        return dtypes_dict


class EvolutionaryCombiner(object):
    """Combiner object that produces crossover between two individuals.

    Arguments:
        grid (dict): dictionary mapping hyperparameters to either a two
        element tuple (min, max) for continuous variables or a set of
        values.
        kwargs (dict): dictionary mapping data types to functions that combine
        individuals.
    """

    def __init__(self,
                 grid,
                 kwargs=dict()):
        self.grid = grid
        self._key_to_dtypes = self._get_dtypes(grid)
        self._dtypes_to_func = {
            int: self._combine_int,
            float: self._combine_float,
            str: self._combine_str,
            tuple: self._combine_tuple,
            list: self._combine_list,
            set: self._combine_set}
        self._dtypes_to_func.update(kwargs)

    def combine(self, ind1, ind2):
        for key in ind1.data.keys():
            combination_func = self._dtypes_to_func[self._key_to_dtypes[key]]
            # If is modifiable.
            combination_func(key, ind1, ind2)
        return ind1, ind2

    def _combine_int(self, key, ind1, ind2):
        ind1.data[key], ind2.data[key] = ind2.data[key], ind1.data[key]
        return  None

    def _combine_float(self, key, ind1, ind2):
        return self._combine_int(key, ind1, ind2)

    def _combine_str(self, key, ind1, ind2):
        raise NotImplementedError
        return None

    def _combine_tuple(self, key, ind1, ind2):
        raise NotImplementedError
        return None

    def _combine_list(self, key, ind1, ind2):
        raise NotImplementedError
        return None

    def _combine_set(self, key, ind1, ind2):
        raise NotImplementedError
        return None

    def _get_dtypes(self, grid):
        dtypes_dict = dict()
        for key, value in grid.items():
            for dtype in (int, float, str, tuple, list, set, frozenset):
                if isinstance(random.sample(value, 1)[0], dtype):
                    dtypes_dict[key] = dtype
                    break
            else:  # exhausted loop and found no dtype.
                raise NotImplementedError(
                    'No dtype for {0} was found.'.format(value))
        return dtypes_dict

###############################################################################
# class EvolutionaryEvaluator(object):
#     """Evolutionary evaluator with sklearn_utilities defaults."""
#
#     def __init__(self, evaluator):
#         if combiner is None:
#             self.combiner = self._get_population(self)
#         if mutator is None:
#             self.population = self._get_population(self)
