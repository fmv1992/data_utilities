"""Test sklearn_utilities from this module."""
import glob
import multiprocessing
import os
import warnings
import datetime as dt
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from data_utilities import sklearn_utilities as su
from data_utilities.tests.test_support import (
    TestDataUtilitiesTestCase,
    TestMetaClass,
    TestSKLearnTestCase,
    time_function_call)


class TestGridSearchCV(TestDataUtilitiesTestCase, metaclass=TestMetaClass):
    """Test class for persistent_grid_search_cv of sklearn_utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up class method from unittest.

        Initialize:
            * Data to be used on tests.
            * Support directories.

        """
        # Create data.
        cls.data_ml_x, cls.data_ml_y = datasets.make_hastie_10_2(
            n_samples=cls.n_lines_test_sklearn, random_state=1)
        cls.small_grid = {'n_estimators': list(range(40, 44)),
                          'max_depth': [4, 9],
                          'min_samples_leaf': [.2],
                          'n_jobs': [1, ],
                          }
        # Call parent class super.
        super().setUpClass()
        # Create support directories.
        cls.test_directory_grid_search_data = os.path.join(
            cls.test_directory.name, 'data')
        os.mkdir(cls.test_directory_grid_search_data)
        # Create a csv file.
        cls.csv_path = os.path.join(cls.test_directory_grid_search_data,
                                    'data.csv')
        pd.concat(map(pd.DataFrame,
                      (cls.data_ml_x, cls.data_ml_y)),
                  axis=1).to_csv(cls.csv_path)

    def setUp(self):
        """Set up method from unittest.

        Filter warnings of the type: UserWarning.

        """
        super().setUp()
        # Ignore joblib/parallel.py if using threads.
        if os.name == 'nt':
            warnings.filterwarnings('ignore', category=UserWarning)

    def tearDown(self):
        """Tear down method from unittest.

        Compute and display running time for test method.

        """
        all_pickle_files = (
            glob.glob(os.path.join(self.test_directory.name,
                                   '**.pickle'))
            + glob.glob(os.path.join(self.test_directory.name,
                                     '**' + os.sep + '*.pickle')))
        for remove_pickle in all_pickle_files:
            os.remove(remove_pickle)
        # os.system('tree ' + self.test_directory.name)
        # Restore warnings.
        if os.name == 'nt':
            warnings.filterwarnings('default', category=UserWarning)

    def test_multiparallelism_speed(self):
        """Test that using more processes speed up the grid search."""
        clf = RandomForestClassifier()

        all_times = []
        N_RUNS = 3
        cpu_count = multiprocessing.cpu_count()
        for processors in (1, cpu_count):
            processor_times = []
            for _ in range(N_RUNS):
                # Decorate persistent_grid_search_cv.
                time_func = time_function_call(su.persistent_grid_search_cv)
                run_time = time_func(
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
        # Windows...
        if os.name == 'nt':
            all_times_norm = all_times / all_times.max()
            # Assert that all times are very similar.
            assert(all(all_times_norm > 0.9))
        else:
            assert all(np.diff(all_times) < 0)

    def test_grid_search(self):
        """Simplest test to persistent_grid_search_cv function."""
        # Initiate a persistent grid search.
        bpg1 = su.grid_search.PersistentGrid(
            persistent_grid_path=os.path.join(self.test_directory.name,
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
            persistent_grid_path=os.path.join(self.test_directory.name,
                                              'bpg.pickle'),
            dataset_path=self.csv_path)
        grid2 = su.persistent_grid_search_cv(  # noqa
            bpg2,
            self.small_grid,
            RandomForestClassifier(),
            self.data_ml_x,
            y=self.data_ml_y,
            cv=10,
            n_jobs=multiprocessing.cpu_count())
        # TODO: assert that the second run is way faster than the first.

    def test_grid_search_times(self):
        """Test that the persistent grid is in fact saving time."""
        # Decorate persistent_grid_search_cv.
        time_func = time_function_call(su.persistent_grid_search_cv)

        # Initialize needed objects.
        clf = RandomForestClassifier()
        bpg1 = su.grid_search.PersistentGrid(
            persistent_grid_path=os.path.join(self.test_directory.name,
                                              'bpg.pickle'),
            dataset_path=self.csv_path)

        # Compute run times.
        first_run_time = time_func(
            bpg1,
            self.small_grid,
            clf,
            self.data_ml_x,
            n_jobs=-1,
            y=self.data_ml_y,
            scoring='roc_auc',
            cv=5)
        second_run_time = time_func(
            bpg1,
            self.small_grid,
            clf,
            self.data_ml_x,
            n_jobs=-1,
            y=self.data_ml_y,
            scoring='roc_auc',
            cv=5)
        MIN_SLOW_TO_FAST_RATIO = 10
        assert first_run_time/second_run_time > MIN_SLOW_TO_FAST_RATIO

    def test_grid_search_times_are_similar(self):
        """Test that the persistent grid called twice has similar times."""
        # Define grids with virtually the same running time.
        grid1 = self.small_grid.copy()
        grid1['random_state'] = [1, ]
        grid2 = self.small_grid.copy()
        grid2['random_state'] = [0, ]

        # Decorate persistent_grid_search_cv.
        time_func = time_function_call(su.persistent_grid_search_cv)

        # Initialize needed objects.
        clf = RandomForestClassifier()
        bpg1 = su.grid_search.PersistentGrid(
            persistent_grid_path=os.path.join(self.test_directory.name,
                                              'bpg.pickle'),
            dataset_path=self.csv_path)

        # Compute first run time.
        first_run_time = time_func(
            bpg1,
            grid1,
            clf,
            self.data_ml_x,
            n_jobs=-1,
            y=self.data_ml_y,
            scoring='roc_auc',
            cv=5)
        second_run_time = time_func(
            bpg1,
            grid2,
            clf,
            self.data_ml_x,
            n_jobs=-1,
            y=self.data_ml_y,
            scoring='roc_auc',
            cv=5)
        third_run_time = time_func(
            bpg1,
            grid2,
            clf,
            self.data_ml_x,
            n_jobs=-1,
            y=self.data_ml_y,
            scoring='roc_auc',
            cv=5)
        fourth_run_time = time_func(
            bpg1,
            grid2,
            clf,
            self.data_ml_x,
            n_jobs=-1,
            y=self.data_ml_y,
            scoring='roc_auc',
            cv=5)
        # First compare that running with different seeds has almost equal
        # times.
        max_runtime = max((first_run_time, second_run_time))
        min_runtime = min((first_run_time, second_run_time))
        assert min_runtime/max_runtime > 0.9
        # Then compare that those values were in fact stored.
        eq1_runtime = max((third_run_time, fourth_run_time))
        eq2_runtime = min((third_run_time, fourth_run_time))
        assert eq1_runtime - eq2_runtime < dt.timedelta(seconds=0.5)


class TestXGBoostFunctions(TestSKLearnTestCase, metaclass=TestMetaClass):
    """Test class to test all XGBoost related functions."""

    @unittest.skipIf(not HAS_XGBOOST, 'xgboost not present.')
    def test_xgboost_get_learning_curve(self):
        """Test xgboost_get_learning_curve syntax."""
        estimator = XGBClassifier(
            n_estimators=20,
            nthread=4)
        estimator.fit(self.x_train, self.y_train.values.ravel())
        lcurve = su.xgboost_get_learning_curve(
            estimator,
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test)
        # Plot results.
        if self.save_figures:
            fig, ax = plt.subplots()
            y2 = lcurve['train_scores']
            y1 = lcurve['test_scores']
            x = np.arange(len(y1))
            ax.plot(x, y1, label='test')
            ax.plot(x, y2, label='train')
            ax.set_ylim(0, 1)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(
                self.test_directory.name,
                'test_xgboost_get_learning_curve.png'),
                        dpi=300)

    @unittest.skipIf(not HAS_XGBOOST, 'xgboost not present.')
    def test_xgboost_get_feature_importances_from_booster(self):
        """Test xgboost_get_feature_importances_from_booster syntax."""
        estimator = XGBClassifier(n_estimators=10, nthread=4)
        estimator.fit(self.x_train, self.y_train.values.ravel())
        fi = su.xgboost_get_feature_importances_from_booster(
            estimator.get_booster())
        assert isinstance(fi, pd.DataFrame)
