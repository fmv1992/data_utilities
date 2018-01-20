"""Offer an evolutionary persistent parallel grid search functionality.

This library offers a genetic/evolutionary approach for hyper parameter grid
search in the marchine learning context (1). It also stores the evolution
data (2), provides means of paralellization (3) and gives sane defaults (4).

(1): It offers the simple genetic algorithm of cross-over and mutation
(easimple from deap) giving the user freedom to define their own mating and
mutating functions. It also takes bounds for each parameter and clips results
that are outside the allowed range.

(2): TODO

(3): TODO

(4): TODO

Classes:
    EvolutionaryPersistentGrid: pickled object that stores evolution states.
    EvolutionaryPersistentGridSearchCV: scikit-learn-like object.
    EvolutionaryToolbox: analogue of deap's toolboxes.
    IndividualFromGrid: analogue of deap's individual.
    BasePersistentEvolutionaryOperator: input to EToolbox.
    EvolutionaryMutator: input to EToolbox.
    EvolutionaryCombiner: input to EToolbox.

EvolutionaryPersistentGrid:
* Directly saves the necessary data to resume an evolution. The important data
is the population and the best hyperparameters.
* All saved data is *solely* stored here.

EvolutionaryPersistentGridSearchCV
* Scikit-learn-like object with a fit and predict methods. Attributes:
best_params, best_estimator_ and best_score_.

EvolutionaryToolbox
* Correspondent to deap's toolbox. Thus it stores the mate, mutate, select,
population and evaluate methods.
* TODO: Comment on API requirements? Evaluate and so on.

IndividualFromGrid
* Holds the hyper parameters values in their data attributes and are
combined/mutated/selected during evolution.

BasePersistentEvolutionaryOperator
* Base class for EvolutionaryMutator and EvolutionaryCombiner.

EvolutionaryMutator
* Object that stores different mutation function for each parameter.
* Has a set defaults out of the box in case functions are not specified for
each parameter.
* TODO: Comment on API requirements? Evaluate and so on.

EvolutionaryCombiner
* Object that stores different combination functions for each parameter.
* Has a set defaults out of the box in case functions are not specified for
each parameter.
* TODO: Comment on API requirements? Combine and so on.

This module depends on deap.

This script follows the guidelines of the sklearn project:
http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects

The idea is that the EvolutionaryPersistentGridSearchCV object contains a an
attribute that is the persistent evolutionary object.

Its fit method would call an evolutionary function that would store
intermediate results in the persistent evolutionary object.

TODO:
    * Clone the estimators instead of refit them every time (error prone). This
    is what scikit-learn do.

"""

import copy
import hashlib
import itertools
import os
import random
import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import deap.base
import deap.creator
import deap.tools
from deap.algorithms import eaSimple

from data_utilities.sklearn_utilities import _get_n_jobs_as_joblib_parallel
from data_utilities.sklearn_utilities.grid_search import BasePersistentGrid

# If on windows use threading. Jesus, Windows...
if os.name == 'nt':
    warnings.warn('Mocking multiprocessing capabilities on Windows using '
                  'threading.',
                  UserWarning)
    import multiprocessing.dummy as mp
else:
    import multiprocessing as mp

# pylama: ignore=D103,D102,W0611

# Global objects to be assigned to as the data.
_X_MATRIX = None
_Y_ARRAY = None


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

    Arguments: TODO.

    Example:
        >>> TODO

    """

    # TODO: allow for statistics object (deap).
    def __init__(self,
                 ev_func,
                 ef_args=tuple(),
                 ef_kwargs=dict(),
                 persistent_grid_path=None,
                 dataset_path=None,
                 hash_function=hashlib.md5,
                 save_every_n_interations=10):
        """Initialize EvolutionaryPersistentGrid object."""
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
        self._ngen_count = 0
        # Create a dictionary of the form:
        # {'argname': passed_value, ..., 'argname': default_if_non_passed}
        # Base hash:
        # The base hash is the hash that is combined with any new
        # parameters go guarantee that the combination of
        # hash(base_hash + hash(parameter)) is unique.
        # For the case of the EvolutionaryPersistentGrid the base hash is the
        # hash(hash(dataset) + hash(arg1) + hash(arg2) + ...
        # + hash(function name).

        # Python3.4 does not allow for unpacking ef_args inside a tuple in
        # multiline.
        self.hash_sequence = ((dataset_path,) + tuple(ef_args) +
                              (ef_kwargs, ev_func.__name__,))
        self.base_hash = self.get_hash(b''.join(
            map(self.get_hash, map(self._transform_to_hashable,
                                   self.hash_sequence))))

        # TODO: unambiguously determine that population should be refreshed and
        # stored here upon pickling and that it shall be transfered back to
        # toolbox upon unpickling.


# TODO: clarify who sets what (toolbox, epgscv, etc).
class EvolutionaryPersistentGridSearchCV(BaseEstimator, ClassifierMixin):
    """Perform an evolutionary grid search.

    Perform an evolutionary grid search with cross validation and persistence.

    Arguments: TODO.

    Example:
        >>> TODO

    """

    def __init__(self,
                 persistent_evolutionary_object,
                 estimator, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
        """Initialize EvolutionaryPersistentGridSearchCV object."""
        super().__init__()
        self.n_jobs = n_jobs

        self.epgo = persistent_evolutionary_object
        self.epgo.toolbox.estimator = estimator
        if hasattr(self.epgo, '_best_score'):
            self._update_from_epgo()

    def _update_from_epgo(self):
        self.best_score_ = self.epgo._best_score
        self.best_params_ = self.epgo._best_params
        self.epgo.toolbox.estimator.set_params(**self.best_params_)
        if hasattr(self, 'x_'):
            self.epgo.toolbox.estimator.fit(self.x_, self.y_,)
            self.best_estimator_ = self.epgo.toolbox.estimator
        else:
            self.best_estimator_ = None

    def fit(self, x, y):
        """Fit evolutionary object.

        Fit the evolutionary object saving the best results in the
        EvolutionaryPersistentGrid object attribute.

        """
        # Check that x and y have correct shape and does not contain nans.
        # x, y = check_X_y(x, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Store the data.
        self.x_ = x
        self.y_ = y

        global _X_MATRIX
        global _Y_ARRAY
        _X_MATRIX = self.x_
        _Y_ARRAY = self.y_

        # Iterate over populations.
        self._evolve()

        # Save best parameters.
        self._save_best()

        # Return the classifier
        return self

    def _evolve(self):
        # Make individuals point to x and y.
        with mp.Pool(_get_n_jobs_as_joblib_parallel(self.n_jobs)) as mp_pool:
            self.epgo.toolbox.register('map', mp_pool.map)
            # Execute evolution cycles.
            while self.epgo._ngen_count < self.epgo.ngen:
                if self.epgo.save_every_n_interations < (
                            self.epgo.ngen - self.epgo._ngen_count):
                    nruns = self.epgo.save_every_n_interations
                else:
                    nruns = self.epgo.ngen - self.epgo._ngen_count
                self.epgo.pop, logbook = eaSimple(
                    self.epgo.pop,
                    self.epgo.toolbox,
                    self.epgo.cxpb,
                    self.epgo.mutpb,
                    nruns,
                    **self.epgo.ef_kwargs)
                self.epgo.save()
                self.epgo._ngen_count += nruns
        return None

    def _save_best(self):
        _best_ind = deap.tools.selBest(self.epgo.pop, 1)[0]
        _best_score = _best_ind.fitness.values[0]
        if _best_score >= getattr(self.epgo, '_best_score', float('-inf')):
            self.epgo._best_score = _best_score
            self.epgo._best_params = _best_ind.data.copy()
            self._update_from_epgo()

    def predict(self, X):
        """Predict input data."""
        # TODO: implement this function.
        raise NotImplementedError
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


class EvolutionaryToolbox(deap.base.Toolbox):
    """Evolutionary toolbox with sklearn_utilities defaults.

    Arguments: TODO.

    Example:
        >>> TODO

    """

    def __init__(self,
                 grid,
                 grid_bounds=dict(),
                 combiner=None,
                 mutator=None,
                 population=None,
                 cross_val_score_kwargs=dict(),  # maybe required.
                 cross_val_score_aggr_function=np.mean,
                 ):
        """Initialize EvolutionaryToolbox object.

        TODO: comment on initialization and flexibility. Evaluation? Scoring?
        How customize?

        Arguments:
            scoring (TODO): argument to sklearn.get_scorer.
        """
        self.grid = grid
        self.grid_bounds = grid_bounds
        self.cross_val_score_aggr_function = cross_val_score_aggr_function

        super().__init__()

        if combiner is None:
            self.combiner = EvolutionaryCombiner(grid, grid_bounds=grid_bounds)
        else:
            self.combiner = combiner
        if mutator is None:
            self.mutator = EvolutionaryMutator(grid, grid_bounds=grid_bounds)
        else:
            self.mutator = mutator
        self.cross_val_score_kwargs = cross_val_score_kwargs.copy()
        self.cross_val_score_kwargs['scoring'] = cross_val_score_kwargs.get(
            'scoring', 'roc_auc')

        self.register('mutate', mutator.mutate)
        self.register('combine', combiner.combine)
        self.mate = self.combine
        self.register('select', deap.tools.selTournament, tournsize=3)

        # TODO: cover both cases of None and not None.
        if isinstance(population, int):
            self.pop = self._create_population(grid, population)
        else:
            self.pop = population

    def _create_population(self, grid, n_individuals):
        # All sklearns scorer objects follow the conveion of
        # 'greater_is_better'.
        deap.creator.create('FitnessMax',
                            deap.base.Fitness,
                            weights=(1, ))
        return list(map(IndividualFromGrid, itertools.repeat(grid,
                                                             n_individuals)))

    # TODO: add flexibility in evaluation function.
    # TODO: add support for a different aggregator than cross_val_score.
    def evaluate(self, individual):
        """Evaluate the individual fitness."""
        global _X_MATRIX
        global _Y_ARRAY
        self.estimator.set_params(**individual.data)
        self.estimator.fit(_X_MATRIX, _Y_ARRAY)
        return self.cross_val_score_aggr_function(
            cross_val_score(self.estimator, _X_MATRIX, _Y_ARRAY,
                            **self.cross_val_score_kwargs)),   # As tuple.

    def __getstate__(self):
        """Remove multiprocessing objects and faulty use of decorators."""
        self_dict = self.__dict__.copy()
        # Delete unpickable multiparallel related objects.
        del self_dict['map']
        return self_dict

    def __setstate__(self, state):
        """Restore multiprocessing attributes."""
        self.__dict__.update(state)


class IndividualFromGrid(object):
    """Individual object to be created from grid."""

    def __init__(self,
                 grid):
        """Create an individual to evolve from a grid dictionary.

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
        self.fitness = deap.creator.FitnessMax()

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
        if all(map(lambda x:isinstance(x, int), arg_tup)):
            return random.randint(*arg_tup)
        elif all(map(lambda x:isinstance(x, float), arg_tup)):
            return random.uniform(*arg_tup)
        else:
            raise ValueError(
                'Tuple {0} does not have either uniform integer types nor '
                'uniform float types.')

    @classmethod
    def _init_from_set(cls, arg_set):
        return random.sample(arg_set, 1)[0]

    def __getstate__(self):
        """Remove multiprocessing objects and faulty use of decorators."""
        self_dict = copy.deepcopy(self.__dict__)
        # TODO: this block if uncommented removes x and y from evaluation.
        # if 'x' in self_dict.keys():
        #     del self_dict['x']
        # if 'y' in self_dict.keys():
        #     del self_dict['y']
        return self_dict


class BasePersistentEvolutionaryOperator(object):
    """Base class for combinator, mutator, etc operators.

    Arguments:
        grid (dict): dictionary mapping hyperparameters to either a two
        element tuple (min, max) for continuous variables or a set of
        values.
        grid_bounds (dict): TODO.
        kwargs (dict): dictionary mapping data types to functions that mutate
        individuals.

    Arguments: TODO.

    Example:
        >>> TODO

    """

    def __init__(self,
                 grid,
                 grid_bounds,
                 params_to_funcs):
        """Initialize BasePersistentEvolutionaryOperator object."""
        self.grid = grid
        self.grid_bounds = grid_bounds
        self._key_to_dtypes = self._get_dtypes(grid)

        for key in grid.keys():
            specific_func = params_to_funcs.get(key, None)
            if specific_func is None:
                specific_func = self._dtypes_to_func[self._key_to_dtypes[key]]
            setattr(self,
                    self._get_func_name_from_param(key),
                    specific_func)

    def _get_func_name_from_param(self, param):
        return self._dynamic_function_prefix + '_' + param

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

    def _bound_function(self, result, bounds):
        if isinstance(bounds, frozenset):
            assert result in bounds
            return result
        elif isinstance(bounds, tuple):
            lower, upper = bounds
            if result < lower:
                return lower
            elif result > upper:
                return upper
        else:
            raise ValueError(
                'Got {0} of type {1} but expecter either a frozenset or '
                'a tuple.'.format(bounds, type(bounds)))
        return result


class EvolutionaryMutator(BasePersistentEvolutionaryOperator):
    """Mutator object that mutates individuals from grid.

    TODO: add explanation.

    Arguments:
        grid (dict): dictionary mapping hyperparameters to either a two
        element tuple (min, max) for continuous variables or a set of
        values.
        kwargs (dict): dictionary mapping data types to functions that mutate
        individuals.

    Example:
        >>> TODO

    """

    def __init__(self,
                 grid,
                 grid_bounds=dict(),
                 params_to_funcs=dict()):
        """Initialize EvolutionaryMutator object."""
        self._dynamic_function_prefix = '_mutate'
        self._dtypes_to_func = {
            int: self._mutate_int,
            float: self._mutate_float,
            str: self._mutate_str,
            tuple: self._mutate_tuple,
            list: self._mutate_list,
            set: self._mutate_set, }
        super().__init__(grid, grid_bounds, params_to_funcs)

    def mutate(self, individual):
        """Mutate individual returning a tuple (individual, ).

        Indivual's parameters are changed inplace.

        """
        for key in individual.data.keys():
            mutated_value = getattr(
                self,
                self._get_func_name_from_param(key))(individual.data[key])
            mutated_value = self._bound_function(mutated_value,
                                                 self.grid_bounds[key])
            if mutated_value is not None:  # That means object is not mutable.
                individual.data[key] = mutated_value
        return (individual, )

    def _mutate_int(self, value):
        """Mutate integers mostly between +2 and -2."""
        increment = int(np.random.normal(loc=0, scale=2))
        while increment == 0:
            increment = int(np.random.normal(loc=0, scale=2))
        return value + increment

    def _mutate_float(self, value):
        """Mutate integers mostly between +2.5 and -2.5."""
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


class EvolutionaryCombiner(BasePersistentEvolutionaryOperator):
    """Combiner object that produces crossover between two individuals.

    Arguments:
        grid (dict): dictionary mapping hyperparameters to either a two
        element tuple (min, max) for continuous variables or a set of
        values.
        kwargs (dict): dictionary mapping data types to functions that combine
        individuals.

    Arguments: TODO.

    Example:
        >>> TODO

    """

    def __init__(self,
                 grid,
                 grid_bounds=dict(),
                 params_to_funcs=dict()):
        """Initialize EvolutionaryCombiner object."""
        self._dynamic_function_prefix = '_combine'
        self._dtypes_to_func = {
            int: self._combine_int,
            float: self._combine_float,
            str: self._combine_str,
            tuple: self._combine_tuple,
            list: self._combine_list,
            set: self._combine_set}
        super().__init__(grid, grid_bounds, params_to_funcs)

    def combine(self, ind1, ind2):
        """Combine two individuals returning a tuple (ind1, ind2).

        Indivual's parameters are changed inplace.

        """
        for key in ind1.data.keys():
            new1, new2 = getattr(
                self,
                self._get_func_name_from_param(key))(key, ind1, ind2)
            # TODO: bound part has a set as bounds. Maybe it is not necessary
            # to bound mating?
            # print('bound:', self.grid_bounds[key])
            # new1 = self._bound_function(new1,
            #                             *self.grid_bounds[key])
            # new2 = self._bound_function(new2,
            #                             *self.grid_bounds[key])
            ind1.data[key], ind2.data[key] = new1, new2
        return (ind1, ind2)

    def _combine_int(self, key, ind1, ind2):
        new1, new2 = ind2.data[key], ind1.data[key]
        return (new1, new2)

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
