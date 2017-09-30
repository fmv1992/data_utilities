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

from data_utilities import sklearn_utilities as su

class BasePersistentGrid(object):
    def __new__(cls, *args, **kwargs):
        """Allow single interface for creating and loading objects:

        Example:
            >>> path = '/path/to/persistent/file'
            >>> persistent_grid = (
                grid_search.BasePersistentGrid.load_from_path(path))
        """
        loaded_object = cls.load_from_path(*args, **kwargs)
        if loaded_object is None:  # Intialize with __init__ method.
            # Will call init and set all attributes (including data set's).
            return super(BasePersistentGrid, cls).__new__(cls)
        else:
            return loaded_object

    def __init__(self,
                 persistence_grid_path=None,
                 dataset_path=None,
                 hash_function=hashlib.md5,
                 save_every_n_interations=1):
            # Create creation date if needed.
            self.lock = None
            self.temporary_results = {}

            # These attributes are constant even between runs.
            self.hash_function = hash_function
            self.persistence_grid_path = persistence_grid_path
            self.save_every_n_interations = save_every_n_interations

            # These two attributes can change between interactions.
            self.dataset_path = dataset_path
            self.dataset_hash = (
                self.dataset_hash if hasattr(self, 'dataset_hash')
                else self._update_data_set_hash(self.dataset_path))

            # These attributes change every session.
            self._n_counter = 0

    @classmethod
    def load_from_path(cls, *args, **kwargs):
        if kwargs:
            path = kwargs['persistence_grid_path']
        else:
            return None

        if os.path.isfile(path):
            with open(path, 'rb') as f:
                loaded_object = pickle.load(f)
            # Refresh data set hash. All other attributes are kept.
            loaded_object._update_data_set_hash(kwargs['dataset_path'])
            return loaded_object
        else:
            return None

    def _update_data_set_hash(self, dataset_path):
        return self.get_hash_from_path(dataset_path)

    def update(self, estimator, results):
        # Get A = hash(hash(dataset) + hash(estimator)).
        # Get B = hash(results).
        # Store results in temporary_results[A][B].
        # Store latest updated time.
        # If needed write it to ./A/B.pickle.
        pass

    def retrieve(self, estimator, grid):
        # If already stored just return.
        # Else calculate, update and return.
        pass

    def get_hash_from_path(self, path):
        # Tested ;)
        hash_obj = self.hash_function()
        with open(path, 'rb') as f:
            data = f.read(io.DEFAULT_BUFFER_SIZE)
            while data:
                hash_obj.update(data)
                data = f.read(io.DEFAULT_BUFFER_SIZE)
        return hash_obj.digest()

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

if __name__ == '__main__':
    bpg = PersistentGrid(
        '../../__/persistent_grid.pickle',
        '../../__/iris_dataset.csv')
