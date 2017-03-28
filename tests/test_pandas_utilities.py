"""Test pandas_utilities from this module."""

import numpy as np
import pandas as pd
import itertools
import random
import unittest
import pandas_utilities as pu
from test_support import TestMetaClass

N = 1000
l = 1000
c = 20

class FindComponentsOfArray(unittest.TestCase, metaclass=TestMetaClass):  # noqa

    """Test class for function of pandas_utilities.

    The interface should make easy to specify:
        - N: the quantity of tests to be run.
        - l: the length of the arrays.
        - c: the number of columns to generate each y.

    The tests should cover the case of integer and non integer composition of
    arrays.

    The tests should cover the case of arrays equal to the result and equal to
    compositions of other columns.

    The tests should cover the case where there is no possible composition.

    """

    @classmethod
    def setUpClass(cls):
        """Setup attributes once."""
        cls.N = N
        cls.l = l
        cls.c = c
        cls.data = pu.statistical_distributions_dataframe(
            shape=(cls.l, cls.c))

    def setUp(self):
        """setUp method from unittest."""
        pass

    def gen_multipliers(self, n=N, amplitude=1500):
        """Create a tuple of multipliers."""
        MULT_AMPLITUDE = 1500
        multipliers_pool = np.concatenate((
            np.arange(1, MULT_AMPLITUDE + 1),
            np.arange(- MULT_AMPLITUDE, -1 + 1)))

        # Create all the multipliers once in memory.
        multipliers = tuple(
            np.random.choice(multipliers_pool, size=self.c)
            * np.random.randint(0, 1+1, size=self.c, dtype=bool)
            for x in range(N))

        return multipliers

    def gen_y_from_multipliers(self, multipliers):
        """Create an iterator of y arrays from multipliers."""
        mask = (multipliers != 0)

        y = (self.data.iloc[:, mask] * multipliers[mask]).sum(axis=1)

        return y

    def dict_results_from_multipliers(self, multipliers):
        """Create an iterator of dictionaries with answers from multipliers."""
        mask = (multipliers != 0)
        return dict(zip(self.data.columns[mask], multipliers[mask]))

    def test_integer_composition(self):
        """integer composition test."""
        # Setup variables
        multipliers = self.gen_multipliers()

        # Map the multipliers to a generator of ys.
        y_map = map(self.gen_y_from_multipliers, multipliers)

        # Map the multipliers to a generator of dictionaries.
        dicts_map = map(self.dict_results_from_multipliers, multipliers)

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


class UtilitiesDataFrames(unittest.TestCase, metaclass=TestMetaClass):  # noqa

    """Test class for functions that create 'out of the shelf' dataframes.

    The interface should make easy to specify:
        - N: the quantity of tests to be run.
        - l: the length of the arrays.
        - c: the number of columns to generate each y.

    The tests should cover issues from extreme cases cases of inputs as well as
    different datatypes.

    """

    def setUp(self):
        """setUp method from unittest."""
        pass

    def test_dummy_dataframe(self):
        """dummy dataframe test."""
        # Function should work 'out of the box'.
        pu.dummy_dataframe()

        # First argument should also be the shape.
        pu.dummy_dataframe(100)
        pu.dummy_dataframe((100, 100))

        # Test shapes with integers.
        # Prepare test shapes with integer values.
        integer_shapes = (random.randint(1, l) for x in range(N//2))
        # Prepare test shapes with integer borderline values.
        integer_borderline_shapes = range(0, 1 + 1)
        iterator_of_shapes = itertools.chain(
            integer_shapes, integer_borderline_shapes)
        map_dummy_dataframe = map(lambda x: pu.dummy_dataframe(x),
                                  iterator_of_shapes)
        self.assert_X_from_iterables(
            self.assertIsInstance,
            map_dummy_dataframe,
            itertools.repeat(pd.DataFrame))
        del (integer_shapes, integer_borderline_shapes, iterator_of_shapes,
             map_dummy_dataframe)

        # Test shapes with tuples.
        # Prepare test shapes with tuple values.
        tuple_shapes = (
            (random.randint(1, l),
             random.randint(1, c)) for x in range(N//2))
        # Prepare test shapes with tuple borderline values.
        borderline_value1 = (1, 4)
        borderline_value2 = (0, 4)
        tuple_borderline_shapes = (borderline_value1, borderline_value2)
        iterator_of_shapes = itertools.chain(
            tuple_shapes, tuple_borderline_shapes)
        map_dummy_dataframe = map(lambda x: pu.dummy_dataframe(x),
                                  iterator_of_shapes)
        self.assert_X_from_iterables(
            self.assertIsInstance,
            map_dummy_dataframe,
            itertools.repeat(pd.DataFrame))

    def test_statistical_distributions_dataframe(self):
        """statistical distributions dataframe test."""
        # Function should work 'out of the box'.
        pu.statistical_distributions_dataframe()

        # First argument should also be the shape.
        pu.statistical_distributions_dataframe(100)
        pu.statistical_distributions_dataframe((100, 100))

        # Test shapes with integers.
        # Prepare test shapes with integer values.
        integer_shapes = (random.randint(1, l) for x in range(N//2))
        # Prepare test shapes with integer borderline values.
        integer_borderline_shapes = range(0, 1 + 1)
        iterator_of_shapes = itertools.chain(
            integer_shapes, integer_borderline_shapes)
        map_dummy_dataframe = map(
            lambda x: pu.statistical_distributions_dataframe(x),
            iterator_of_shapes)
        self.assert_X_from_iterables(
            self.assertIsInstance,
            map_dummy_dataframe,
            itertools.repeat(pd.DataFrame))
        del (integer_shapes, integer_borderline_shapes, iterator_of_shapes,
             map_dummy_dataframe)

        # Test invalid shapes with tuple.
        # Prepare test shapes with tuple values.
        tuple_shapes = tuple(
            (random.randint(1, l),
             random.randint(1, c)) for x in range(N//2))
        # Prepare test shapes with tuple values.
        borderline_value1 = (1, 4)
        borderline_value2 = (0, 4)
        iterator_of_shapes = (borderline_value1, borderline_value2)
        # Prepare test shapes with tuple borderline values.

        tuple_borderline_shapes = (borderline_value1, borderline_value2)
        iterator_of_shapes = itertools.chain(
            tuple_shapes, tuple_borderline_shapes)
        map_dummy_dataframe = map(
            lambda x: pu.statistical_distributions_dataframe(x),
            iterator_of_shapes)
        self.assert_X_from_iterables(
            self.assertIsInstance,
            map_dummy_dataframe,
            itertools.repeat(pd.DataFrame))
