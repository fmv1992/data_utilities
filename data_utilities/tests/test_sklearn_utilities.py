"""Test sklearn_utilities from this module."""

import os
import datetime as dt
import multiprocessing

import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from data_utilities import sklearn_utilities as su
from data_utilities.tests.test_support import (
    TestDataUtilitiesTestCase, TestMetaClass)


def time_function_call(func, *func_args, **func_kwargs):
    """Time a function call."""
    time_before = dt.datetime.now()
    func(*func_args, **func_kwargs)
    return (dt.datetime.now() - time_before).total_seconds()


class TestGridSearchCV(TestDataUtilitiesTestCase, metaclass=TestMetaClass):
    """Test class for grid_search_cv of sklearn_utilities."""

    data_ml_x, data_ml_y = datasets.make_hastie_10_2(
        n_samples=60000, random_state=1)
    small_grid = {'n_estimators': list(range(1, 3)),
                  'max_depth': [2, 4, 6],
                  'min_samples_leaf': [.2],
                  'n_jobs': [1, ],
                  }

    @classmethod
    def setUpClass(cls):
        # Call parent class super.
        super(TestGridSearchCV, cls).setUpClass()
        # Create support directories.
        cls.temp_directory_grid_search = os.path.join(cls.temp_directory.name,
                                                      'test_grid_search_cv')
        os.mkdir(cls.temp_directory_grid_search)
        cls.temp_directory_grid_search_data = os.path.join(
            cls.temp_directory_grid_search, 'data')
        os.mkdir(cls.temp_directory_grid_search_data)
        # Save data.
        cls.csv_path = os.path.join(cls.temp_directory_grid_search_data,
                                    'data.csv')
        pd.concat(map(pd.DataFrame,
                      (cls.data_ml_x, cls.data_ml_y)),
                  axis=1).to_csv(cls.csv_path)




    # def test_multiparallelism_speed(self):
    #     """Test that using more processes speed up grid search."""
    #     clf = RandomForestClassifier()
    #     random_forest_grid = {
    #         'n_estimators': list(range(1, 3)),
    #         'max_depth': [2, 4, 6],
    #         'min_samples_leaf': [.2],
    #     }

    #     all_times = []
    #     N_RUNS = 3
    #     for processors in (1, multiprocessing.cpu_count()):
    #         processor_times = []
    #         for _ in range(N_RUNS):
    #             processor_times.append(
    #                 time_function_call(
    #                     su.grid_search_cv,
    #                     random_forest_grid,
    #                     clf,
    #                     self.data_ml_x,
    #                     n_jobs=processors,
    #                     y=self.data_ml_y,
    #                     persistence_path=None,
    #                     scoring='roc_auc',
    #                     cv=5))
    #         all_times.append(np.mean(processor_times))
    #     all_times = np.array(all_times)
    #     assert all(np.diff(all_times) < 0)
    def test_grid_search(self):
        persistent_grid_obj = su.grid_search.BasePersistentGrid(
            persistence_grid_path=os.path.join(self.temp_directory_grid_search,
                                               'bpg.pickle'),
            dataset_path=self.csv_path)

        # Do a first run.
        grid = su.grid_search_cv(
            persistent_grid_obj,
            self.small_grid,
            RandomForestClassifier(),
            self.data_ml_x,
            y=self.data_ml_y,
            n_jobs=1)
        import pprint
        pprint.pprint(persistent_grid_obj)
        pprint.pprint(dict(persistent_grid_obj.mp_data))
        os.system('tree '+ self.temp_directory.name)
        p2 = su.grid_search.BasePersistentGrid(
            persistence_grid_path=os.path.join(self.temp_directory_grid_search,
                                               'bpg.pickle'),
            dataset_path=self.csv_path)
