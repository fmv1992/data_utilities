"""Grid search utilities for scikit-learn models.

This module:
    (1) Do one thing and do it well.
    (2) Maximum flexibility, sane defaults.
    (3) Enable paralellism.
    (3) Enable persistence.

"""
import functools
import hashlib
import io
import multiprocessing as mp
import os
import pickle

from data_utilities.sklearn_utilities import get_estimator_name


class BasePersistentGrid(object):
    """Base class for persistent grids.

    It should **always** be initialized using the `load_from_path` constructor.

    Its characteristics are:
    * Simple usage. Instatiation and loading from persistence with the same
    interface (load_from_path).
    * Store combinations of dataset + classifier + grid.
        * TODO: need to figure out a good way to handle processed data sets.
        * TODO: considering the evolutionary approach which data should be
        hashed?
    * Gives support to parallelism (through proper pickling).

    """

    def __init__(self,
                 *args,
                 persistent_grid_path=None,
                 dataset_path=None,
                 hash_function=hashlib.md5,
                 save_every_n_interations=10):
        """Instantiate a BasePersistentGrid object.

        **It is not meant to be used directly. Use the method load_from_path
        instead.**

        """
        # These attributes are constant even between runs.
        self.hash_function = hash_function
        self.persistent_grid_path = persistent_grid_path
        self.save_every_n_interations = save_every_n_interations

        # These two attributes can change between interactions.
        self.dataset_path = dataset_path

        # Initilize multiprocessing attributes: manager, lock, data and
        # _n_counter.
        self.mp_manager = mp.Manager()
        self.mp_lock = self.mp_manager.Lock()
        self.mp_data = self.mp_manager.dict()
        self._mp_n_counter_value = self.mp_manager.Value(int, 0)
        # TODO: add a way to warn if instance is being created directly or
        # through load_from_path.

    @classmethod
    def load_from_path(cls, *args, **kwargs):
        """Unpickle file from persistent_grid_path.

        In addition to it refresh its base_hash based on base_attribute.

        """
        if os.path.isfile(kwargs['persistent_grid_path']):
            with open(kwargs['persistent_grid_path'], 'rb') as f:
                loaded_object = pickle.load(f)
            return loaded_object
        else:
            created_object = cls(*args, **kwargs)
            return created_object

    def update(self, estimator, grid, results):
        """Update storage of hash -> cv scores."""
        request_hash = self.compute_request_hash(estimator, grid)
        self.mp_data[request_hash] = results
        self._mp_n_counter_value.value += 1
        if self._mp_n_counter_value.value % self.save_every_n_interations == 0:
            self.save()

    def save(self):
        """Save persistent object."""
        self.mp_data = dict(self.mp_data)
        with open(self.persistent_grid_path, 'wb') as f:
            pickle.dump(self, f)

    def __setstate__(self, state):
        """Restore multiprocessing attributes."""
        self.__dict__.update(state)
        self.mp_manager = mp.Manager()
        self.mp_lock = self.mp_manager.Lock()
        self.mp_data = self.mp_manager.dict(self.mp_data)
        self._mp_n_counter_value = self.mp_manager.Value(int, 0)

    def __getstate__(self):
        """Delete multiprocessing attributes."""
        mp_attrs = ('mp_manager', 'mp_lock', '_mp_n_counter_value')
        self_dict = self.__dict__.copy()
        for unwanted_attrs in mp_attrs:
            self_dict.pop(unwanted_attrs)
        return self_dict

    def compute_request_hash(self, estimator, grid):
        """Compute requested hash.

        Compute request hash from a combination of:
            * Base hash
            * Estimator name representation
            * Grid value

        """
        estimator_name = get_estimator_name(estimator)
        estimator_hash = self.get_hash(estimator_name)
        grid_hash = self.get_hash(grid)
        final_hash = self.get_hash(
            self.base_hash + estimator_hash + grid_hash)
        return final_hash

    def retrieve(self, estimator, grid):
        """Retrieve requested hash."""
        # If already stored just return.
        request_hash = self.compute_request_hash(estimator, grid)
        retrieved_hash = self.mp_data.get(request_hash, None)
        return retrieved_hash

    def get_hash(self, x):
        """Get hash from an object.

        Use a particular way for computing hash for different python objects
        data types.

        """
        # For dict, bytes and strings.
        if isinstance(x, (dict, bytes)) or not os.path.isfile(x):
            return self._get_hash_from_hashable(self._transform_to_hashable(x))
        # For files.
        elif os.path.isfile(x):
            hash_obj = self.hash_function()
            iter_of_bytes = open(x, 'rb')
            try:
                data = iter_of_bytes.read(io.DEFAULT_BUFFER_SIZE)
                while data:
                    hash_obj.update(data)
                    data = iter_of_bytes.read(io.DEFAULT_BUFFER_SIZE)
            finally:
                iter_of_bytes.close()
        else:
            raise NotImplementedError(
                'Function get_hash is not implemented for data type'
                ' {0}.'.format(str(type(x))))

        return hash_obj.digest()

    def _transform_to_hashable(self, x):
        if isinstance(x, str):
            bytesobj = x.encode()
        elif isinstance(x, bytes):
            bytesobj = x
        else:
            bytesobj = pickle.dumps(x)
        return bytesobj

    @functools.lru_cache(maxsize=2**12)
    def _get_hash_from_hashable(self, hashable):
        hash_obj = self.hash_function()
        hash_obj.update(hashable)
        return hash_obj.digest()

    def get_multiprocessing_manager(self):
        """Return the multiprocessing manager object."""
        return self.mp_manager

    def _update_base_hash(self, x):
        self.base_hash = self.get_hash(x)


class PersistentGrid(BasePersistentGrid):
    """Allow easy persistence between (possibly interrupted) grid searches.

    Example:
        >>> import pandas as pd, numpy as np
        >>> from sklearn.tree import DecisionTreeClassifier as DTC
        >>> from sklearn.datasets import load_breast_cancer
        >>> pg_path = '/tmp/pg.pickle'
        >>> dset_path = '/tmp/data.csv'
        >>> dataset = load_breast_cancer()
        >>> df = pd.DataFrame(np.column_stack((dataset.data, dataset.target)))
        >>> df.to_csv(dset_path)
        >>> dt_grid = {'split': ['random', 'best'],
        ...            'max_depth': [2, 4]}
        >>> clf = DTC()
        >>> dset_path = '/tmp/data.csv'
        >>> persistent_grid = grid_search.PersistentGrid().load_from_path(
        ...     persistent_grid_path='/tmp/pg.pickle',
        ...     dataset_path='/tmp/data.csv')
        >>> persistent_grid

    """

    def __init__(self,
                 persistent_grid_path=None,
                 dataset_path=None,
                 hash_function=hashlib.md5,
                 save_every_n_interations=10):
        """Instantiate a PersistentGrid object.

        **It is not meant to be used directly. Use the method load_from_path
        instead.**

        """
        super(PersistentGrid, self).__init__(
            persistent_grid_path=persistent_grid_path,
            dataset_path=dataset_path,
            hash_function=hash_function,
            save_every_n_interations=save_every_n_interations)
        # Base hash:
        # The base hash is the hash that is combined with any new
        # parameters go guarantee that the combination of
        # hash(base_hash + hash(parameter)) is unique.
        # For the case of the PersistentGrid the base hash is the dataset.
        # This unique hash will be combined with the hash(classifier) and the
        # hash(parameter)
        self.base_hash = self.get_hash(dataset_path)

    @classmethod
    def load_from_path(cls, *args, **kwargs):
        """Unpickle file from persistent_grid_path.

        In addition to it refresh its base_hash based on base_attribute.

        """
        loaded_object = super(cls, PersistentGrid).load_from_path(*args,
                                                                  **kwargs)
        # For PersistentGrid the base hash is the dataset path.
        loaded_object._update_base_hash(kwargs['dataset_path'])
        return loaded_object
