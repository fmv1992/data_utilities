"""Test sklearn_utilities from this module."""
import os
import multiprocessing
import glob

import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from deap.algorithms import eaSimple
from deap.base import Toolbox

from data_utilities import sklearn_utilities as su
from data_utilities.tests.test_support import (
    TestDataUtilitiesTestCase, TestMetaClass, time_function_call)


class BaseGridTestCase(TestDataUtilitiesTestCase, metaclass=TestMetaClass):
    """Test class for persistent_grid_search_cv of sklearn_utilities."""

    @classmethod
    def setUpClass(cls):
        # Create data.
        cls.data_ml_x, cls.data_ml_y = datasets.make_hastie_10_2(
            n_samples=cls.n_lines_test_sklearn, random_state=1)
        cls.small_grid = {'n_estimators': list(range(40, 44)),
                          'max_depth': [4, 9],
                          'min_samples_leaf': [.2],
                          'n_jobs': [1, ],
                          }
        # Call parent class super.
        super(BaseGridTestCase, cls).setUpClass()
        # Create support directories.
        cls.temp_directory_grid_search = os.path.join(
            cls.temp_directory.name, 'test_persistent_grid_search_cv')
        os.mkdir(cls.temp_directory_grid_search)
        cls.temp_directory_grid_search_data = os.path.join(
            cls.temp_directory_grid_search, 'data')
        os.mkdir(cls.temp_directory_grid_search_data)
        # Create a csv file.
        cls.csv_path = os.path.join(cls.temp_directory_grid_search_data,
                                    'data.csv')
        pd.concat(map(pd.DataFrame,
                      (cls.data_ml_x, cls.data_ml_y)),
                  axis=1).to_csv(cls.csv_path)

    def tearDown(self):
        all_pickle_files = (
            glob.glob(os.path.join(self.temp_directory_grid_search,
                                   '**.pickle'))
            + glob.glob(os.path.join(self.temp_directory_grid_search,
                                     '**' + os.sep + '*.pickle')))
        for remove_pickle in all_pickle_files:
            os.remove(remove_pickle)
        # os.system('tree ' + self.temp_directory.name)


class TestGridSearchCV(BaseGridTestCase, metaclass=TestMetaClass):
    """Test class for persistent_grid_search_cv of sklearn_utilities."""

    def test_multiparallelism_speed(self):
        """Test that using more processes speed up the grid search."""
        clf = RandomForestClassifier()

        all_times = []
        N_RUNS = 3
        cpu_count = multiprocessing.cpu_count()
        for processors in (1, cpu_count):
            processor_times = []
            for _ in range(N_RUNS):
                time_func = time_function_call(su.persistent_grid_search_cv)
                run_time = time_func(
                    # TODO: use real values to improve testing scenario
                    # coverage.
                    su.grid_search.PersistentGrid(
                        persistent_grid_path=os.devnull,
                        dataset_path=os.devnull),
                    self.small_grid,
                    clf,
                    self.data_ml_x,
                    n_jobs=processors,
                    y=self.data_ml_y,
                    scoring='roc_auc',
                    cv=5)
                processor_times.append(run_time.total_seconds())
            all_times.append(np.mean(processor_times))
        all_times = np.array(all_times)
        assert all(np.diff(all_times) < 0)

    def test_grid_search(self):
        # Initiate a persistent grid search.
        bpg1 = su.grid_search.PersistentGrid(
            persistent_grid_path=os.path.join(self.temp_directory_grid_search,
                                              'bpg.pickle'),
            dataset_path=self.csv_path)

        # Do a first run.
        grid = su.persistent_grid_search_cv(
            bpg1,
            self.small_grid,
            RandomForestClassifier(),
            self.data_ml_x,
            y=self.data_ml_y,
            cv=10,
            n_jobs=multiprocessing.cpu_count())
        del grid, bpg1

        # Do a second run.
        bpg2 = su.grid_search.PersistentGrid(
            persistent_grid_path=os.path.join(self.temp_directory_grid_search,
                                              'bpg.pickle'),
            dataset_path=self.csv_path)
        grid2 = su.persistent_grid_search_cv(
            bpg2,
            self.small_grid,
            RandomForestClassifier(),
            self.data_ml_x,
            y=self.data_ml_y,
            cv=10,
            n_jobs=multiprocessing.cpu_count())
        # TODO: assert that the second run is way faster than the first.

class TestEvolutionaryPersistentGrid(BaseGridTestCase,
                                     metaclass=TestMetaClass):
    def test_simple(self):
        population = [list(range(10)) for x in range(10)]
        toolbox = Toolbox()
        cxpb = 0.5
        mutpb = 0.05
        ngen = 50
        easimple_args = (population, toolbox, cxpb, mutpb, ngen)

        epg = su.evolutionary_grid_search.EvolutionaryPersistentGrid.load_from_path(
            eaSimple,
            easimple_args,
            persistent_grid_path=os.devnull,
            dataset_path=os.devnull,
        )

