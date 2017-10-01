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
    def __new__(cls, *args, **kwargs):
        """Allow single interface for creating and loading objects:

        Example:
            >>> path = '/path/to/persistent/file'
            >>> persistent_grid = (
                grid_search.BasePersistentGrid.load_from_path(path))
        """
        loaded_object = cls.load_from_path(**kwargs)
        if loaded_object is None:  # Intialize with __init__ method.
            # Will call init and set all attributes (including data set's).
            return super(BasePersistentGrid, cls).__new__(cls)
        else:
            return loaded_object

    def __init__(self,
                 persistence_grid_path=None,
                 dataset_path=None,
                 hash_function=hashlib.md5,
                 save_every_n_interations=10):

            # Create creation date if needed.
            self.temporary_results = {}

            # These attributes are constant even between runs.
            self.hash_function = hash_function
            self.persistence_grid_path = persistence_grid_path
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

    @classmethod
    def load_from_path(cls, *args, **kwargs):
        if kwargs:
            path = kwargs['persistence_grid_path']
        else:
            return None

        if os.path.isfile(path):
            with open(path, 'rb') as f:
                loaded_object = pickle.load(f)
            # Refresh data set hash.
            loaded_object._update_dataset_hash(kwargs['dataset_path'])
            # All other attributes are kept.
            loaded_object._instantiate_shared_attributes()
            return loaded_object
        else:
            return None

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
        with open(self.persistence_grid_path, 'wb') as f:
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
    """Enable easy persistence of calculated values for grid searchs.

    Not be used directly, just instantiated by the user then fed to other
    functions.

    The persistent grid is a (optionally compressed) file that resides in your
    system. It should be one per project. A project may contain multiple data
    sets and multiple models. For a given data set and model combination only
    one calculated grid may exist.

    It must unequivocally store calculated values for a model for a given data
    set.

    It should also be able to store tens of thousands of calculated grids and
    retrieve them quickly if already calculated.

    It must allow for multiple processes to access it.

    A PersitentGrid must have:
        * A unique data set tied to it (checksum).
            * Changing it should raise a warning.
        * A unique classifier tied to it.
            *
    """
    pass

if __name__ == '__main__':
    bpg = PersistentGrid(
        '../../__/persistent_grid.pickle',
        '../../__/iris_dataset.csv')
