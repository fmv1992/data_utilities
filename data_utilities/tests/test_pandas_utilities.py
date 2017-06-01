"""Test pandas_utilities from this module."""

import itertools
import random

import numpy as np
import pandas as pd
import string

from data_utilities import pandas_utilities as pu
from data_utilities.tests.test_support import (
    TestDataUtilitiesTestCase, TestMetaClass)


def setUpModule():
    """Setup TestDataUtilitiesTestCase 'data' attribute.

    Useful if there is a unittest being run.
    """
    TestDataUtilitiesTestCase.update_data()


class TestFindComponentsOfArray(TestDataUtilitiesTestCase,
                                metaclass=TestMetaClass):
    """Test class for find_components_of_array of pandas_utilities.

    The tests should cover the case of integer and non integer composition of
    arrays.

    The tests should cover the case of arrays equal to the result and equal to
    compositions of other columns.

    The tests should cover the case where there is no possible composition.

    """

    @classmethod
    def setUpClass(cls):
        """setUpClass class method from unittest."""
        pass

    def setUp(self):
        """setUp method from unittest."""
        pass

    # TODO: transform in a _method.
    def _gen_multipliers(self, amplitude=1500):
        """Create a tuple of non-zero multipliers with a given amplitude.

        To be used on test_integer_composition method.

        """
        MULT_AMPLITUDE = 1500
        multipliers_pool = np.concatenate((
            np.arange(1, MULT_AMPLITUDE + 1),
            np.arange(- MULT_AMPLITUDE, -1 + 1)))

        # Create all the multipliers once in memory.
        multipliers = tuple(
            np.random.choice(multipliers_pool, size=self.n_columns)
            # TODO: some of these arrays are zero only ; in this case there
            # should be none.
            * np.random.randint(0, 1+1, size=self.n_columns, dtype=bool)
            for x in range(self.n_tests))

        return multipliers

    # TODO: transform in a _method.
    def _gen_y_from_multipliers(self, multipliers):
        """Create an iterator of y arrays from multipliers.

        To be used on test_integer_composition method.

        """
        mask = (multipliers != 0)

        y = (self.data.iloc[:, mask] * multipliers[mask]).sum(axis=1)

        return y

    def _dict_results_from_multipliers(self, multipliers):
        """Create an iterator of dictionaries with answers from multipliers."""
        mask = (multipliers != 0)
        return dict(zip(self.data.columns[mask], multipliers[mask]))

    def test_integer_composition(self):
        """Integer composition test."""
        # Setup variables
        multipliers = self._gen_multipliers()

        # Map the multipliers to a generator of ys.
        y_map = map(self._gen_y_from_multipliers, multipliers)

        # Map the multipliers to a generator of dictionaries.
        dicts_map = map(self._dict_results_from_multipliers, multipliers)

        # Map the y's to a generator of find_components.
        find_components_map = map(
            lambda y: pu.find_components_of_array(self.data,
                                                  y,
                                                  assume_int=True),
            y_map)

        # Test.
        self.assert_X_from_iterables(
            self.assertEqual,
            dicts_map,
            find_components_map)

    def test_known_to_fail(self):
        """TODO."""
        pass


class TestUtilitiesDataFrames(TestDataUtilitiesTestCase,
                              metaclass=TestMetaClass):
    """Test class for functions that create 'out of the shelf' dataframes.

    The tests should cover issues from extreme cases cases of inputs as well as
    different datatypes.

    """

    def setUp(self):
        """setUp method from unittest.

        Initialize random values to be tested and borderline cases.

        """
        # Test shapes with integers.
        # Prepare test shapes with integer values.
        integer_shapes = (
            random.randint(1, self.n_lines) for x in range(self.n_tests//2))
        # Prepare test shapes with integer borderline values.
        integer_borderline_shapes = range(0, 1 + 1)
        iterator_of_ints = itertools.chain(
            integer_shapes, integer_borderline_shapes)
        # Test shapes with tuples.
        # Prepare test shapes with tuple values.
        tuple_shapes = (
            (random.randint(1, self.n_lines),
             random.randint(1, self.n_columns)) for x in
            range(self.n_tests//2))
        # Prepare test shapes with tuple borderline values.
        borderline_value1 = (1, 4)
        borderline_value2 = (0, 4)
        tuple_borderline_shapes = (borderline_value1, borderline_value2)
        iterator_of_tuples = itertools.chain(
            tuple_shapes, tuple_borderline_shapes)

        # Combine all those iterators on a variable.
        # This iterator_of_shapes
        self.iterator_of_shapes = tuple(itertools.chain(iterator_of_ints,
                                                        iterator_of_tuples))

    def _test_off_the_shelf_functions(self, test_function):
        """Test off the shelf functions."""
        # Function should work 'out of the box'.
        test_function()

        # First argument should also be the shape.
        test_function(100)
        test_function((100, 100))

        map_ots_functions = map(lambda x: test_function(x),
                                self.iterator_of_shapes)
        self.assert_X_from_iterables(
            self.assertIsInstance,
            map_ots_functions,
            itertools.repeat(pd.DataFrame))

    def test_dummy_dataframe(self):
        """Dummy dataframe test."""
        self._test_off_the_shelf_functions(
            pu.dummy_dataframe)

    def test_statistical_distributions_dataframe(self):
        """Statistical distributions dataframe test."""
        self._test_off_the_shelf_functions(
            pu.statistical_distributions_dataframe)

    def test_categorical_serie_to_binary_dataframe(self):
        """Categorical serie to binary dataframe test."""
        # Create artificial categorical series.
        # Define shapes.
        borderline_shapes = tuple(range(1, min(self.n_tests, 10)))
        non_borderline_shapes = np.random.randint(
            1,
            self.n_lines + 1,
            size=max(self.n_tests - len(borderline_shapes), 0))
        shapes = tuple(itertools.chain(borderline_shapes,
                                       non_borderline_shapes))
        # Define categories.
        n_categories = map(
            lambda x: np.random.randint(
                1,
                min(x + 1, 20)),
            shapes)
        n_categories = tuple(n_categories)
        categories = tuple(string.ascii_lowercase)
        # Define series.
        all_series = map(
            lambda x, y: pd.Series(
                np.random.choice(
                    np.random.choice(categories, y),
                    size=x),
                dtype='category'),
            shapes,
            n_categories)
        all_series = tuple(all_series)
        # Update categories to categories that actually show up.
        n_categories_show_up = tuple(
            len(x.unique()) for x in all_series)

        # Create the dataframes.
        df_serie_to_bin = tuple(map(
            pu.categorical_serie_to_binary_dataframe,
            all_series))
        # Test the shape.
        self.assert_X_from_iterables(
            self.assertEqual,
            (x.shape[1] for x in df_serie_to_bin),
            n_categories_show_up)
        # Thest the content/data type.
        self.assert_X_from_iterables(
            self.assertTrue,
            ((x.dtypes == np.bool).all() for x in df_serie_to_bin))

    def _create_tuple_of_unique_values(self, n_values):
        """Create a tuple of unique values."""
        categories = tuple(string.ascii_lowercase)
        n_categories = len(categories)

        # Count the number of letters neeeded to create a unique repetition.
        # Maybe there is a better calculation option using log/ln.
        n_repeats = 0
        n_values_hold = n_values
        while n_values_hold > 1:
            n_values_hold /= n_categories
            n_repeats += 1

        return tuple(
            itertools.islice(
                map(''.join,
                    itertools.product(categories, repeat=n_repeats)),
                n_values))

    def test_object_columns_to_category(self):
        """Object columns to category test."""
        # Create series of unique values (to be kept as object columns).
        #
        # As this operation may be expensive, store the maximum object and
        # create a copy of this.

        ## TODO: refactor code to easily create an iterable with borderline
        ## values and non borderline values.
        ## XXX

        ffobject = self._create_tuple_of_unique_values(int(1e5))
        # Create dataframe of non-unique object values (to be transformed into
        # categories).
        # Create dummy non categorical data (to be kept as numbers).
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        raise Exception
