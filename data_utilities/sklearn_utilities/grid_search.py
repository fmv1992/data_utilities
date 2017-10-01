"""Grid search utilities for scikit-learn models.

    This module:
        (1) Do one thing and do it well.
        (2) Maximum flexibility, sane defaults.
        (3) Enable paralellism.
        (3) Enable persistence.

"""
import pickle
import os
import hashlib
import io
import zipfile
import multiprocessing as mp

import functools

from data_utilities import sklearn_utilities as su

class BasePersistentGrid(object):
    """Base class for persistent grids.

    Its characteristics are:
    * Simple usage. Instatiate it once per project (same call) feed into
    the function and let them take care of the work.
    * Store combinations of dataset + classifier + grid.
    * Parallelization.

    """
    def __new__(cls, **kwargs):
        """Allow single interface for instantiation and creation of objects.

        All objects can be created with `cls.load_from_path()`.

        """
        if os.path.isfile(kwargs['persistent_grid_path']):
            loaded_object = cls.load_from_path(**kwargs)
            return loaded_object
        else:
            return super(BasePersistentGrid, cls).__new__(cls)

    def __init__(self,
                 persistent_grid_path=None,
                 dataset_path=None,
                 hash_function=hashlib.md5,
                 save_every_n_interations=10):

            # Create creation date if needed.
            self.temporary_results = {}

            # These attributes are constant even between runs.
            self.hash_function = hash_function
            self.persistent_grid_path = persistent_grid_path
            self.save_every_n_interations = save_every_n_interations

            # These two attributes can change between interactions.
            self.dataset_path = dataset_path
            if not hasattr(self, 'dataset_hash'):
                self._update_dataset_hash(self.dataset_path)

            # Initilize multiprocessing attributes: manager, lock, data and
            # _n_counter.
            self._instantiate_shared_attributes()

            # Zip file object to be written to disk from time to time.
            # self._data_zip = zipfile.ZipFile(io.BytesIO(), mode='x')
            # Unpacked dictionary: hash -> value
            # self.data = self._load_data_from_bytesio()

            # These attributes change every session.
            # self._n_counter = 0

    @staticmethod
    def load_from_path(*args, **kwargs):
        if os.path.isfile(kwargs['persistent_grid_path']):
            with open(kwargs['persistent_grid_path'], 'rb') as f:
                loaded_object = pickle.load(f)
            # Refresh data set hash.
            print(kwargs['dataset_path'])
            loaded_object._update_dataset_hash(kwargs['dataset_path'])
            # All other attributes are kept.
            loaded_object._instantiate_shared_attributes()
            return loaded_object
        else:
            pg = PersistentGrid(**kwargs)
            return pg

    def get_multiprocessing_manager(self):
        return self.mp_manager

    def _instantiate_shared_attributes(self):
        # Common manager (for other common attributes).
        try:
            self.mp_manager
        except AttributeError:
            self.mp_manager = mp.Manager()
        # Common lock.
        self.mp_lock = self.mp_manager.Lock()
        # Shared computed values (data).
        self.mp_data = self.mp_manager.dict()
        # Common attributes update counter (_n_counter).
        self._mp_n_counter_value = self.mp_manager.Value(int, 0)

    def _update_dataset_hash(self, dataset_path):
        self.dataset_hash = self.get_hash(dataset_path)

    def update(self, estimator, grid, results):
        request_hash = self.compute_request_hash(estimator, grid)
        with self.mp_lock:
            self.mp_data[request_hash] = results
            self._mp_n_counter_value.value += 1
            if self._mp_n_counter_value.value % self.save_every_n_interations == 0:
                self.save()

    def _get_multiprocessing_shared_attributes(self):
        pass

    def save(self):
        # Store values.
        (_store_manager, _store_lock, _store_data, _store_counter) = (
            self.mp_manager, self.mp_lock, self.mp_data,
            self._mp_n_counter_value)
        # Delete values.
        del (self.mp_manager, self.mp_lock, self._mp_n_counter_value)
        # Make sure data is pickable.
        self.mp_data = dict(self.mp_data)
        with open(self.persistent_grid_path, 'wb') as f:
            pickle.dump(self, f)
        # Store saved values.
        (self.mp_manager, self.mp_lock, self.mp_data, self._mp_n_counter_value) = (_store_manager, _store_lock, _store_data, _store_counter)

    def compute_request_hash(self, estimator, grid):
        estimator_hash = self.get_hash(su.get_estimator_name(estimator))
        grid_hash = self.get_hash(grid)
        final_hash = self.get_hash(
            self.dataset_hash + estimator_hash + grid_hash)
        return final_hash

    def retrieve(self, estimator, grid):
        # If already stored just return.
        request_hash = self.compute_request_hash(estimator, grid)
        return self.mp_data.get(request_hash, None)

    def get_hash(self, x):
        # TODO: cached for strings and paths.
        # Need to generalize the reading of:
        #   1) file paths
        #   2) small strings
        #   3) python_objects
        if isinstance(x, (dict, bytes)) or not os.path.isfile(x):  # use function cache.
            return self._get_hash_from_hashable(self._transform_to_hashable(x))
        # For files.
        hash_obj = self.hash_function()
        iter_of_bytes = open(x, 'rb')
        try:
            data = iter_of_bytes.read(io.DEFAULT_BUFFER_SIZE)
            while data:
                hash_obj.update(data)
                data = iter_of_bytes.read(io.DEFAULT_BUFFER_SIZE)
        finally:
            iter_of_bytes.close()
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


    def _load_data_from_bytesio(self, zipf):
        """Return a dictionary of hexdigest -> results."""
        keys = map(lambda x: x.strip('.pickle'), zipf.namelist())
        values = map(zipf.read, zipf.namelist())
        # TODO: use a shared dictionary
        return dict(zip(keys, values))


class PersistentGrid(BasePersistentGrid):
    """Allow easy persistence between interrupted grid searches.

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
        >>> persistent_grid = su.grid_search.PersistentGrid().load_from_path( persistent_grid_path='/tmp/pg.pickle', dataset_path='/tmp/data.csv')
        >>> persistent_grid



    """
    pass

if __name__ == '__main__':
    bpg1 = PersistentGrid(
        persistent_grid_path='../../__/persistent_grid.pickle',
        dataset_path='../../__/iris_dataset.csv')
    bpg2 = PersistentGrid.load_from_path(
        persistent_grid_path='../../__/persistent_grid.pickle',
        dataset_path='../../__/iris_dataset.csv')
