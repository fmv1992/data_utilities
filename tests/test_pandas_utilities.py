"""Test pandas_utilities from this module."""


# import pprint
import pandas as pd
import itertools
import random
import unittest
import pandas_utilities as pu
from test_metaclass import TestMetaClass

N = 1000
l = 1000
c = 20


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
        pu.dummy_dataframe((100, 4))

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

        # Test invalid shapes with tuple.
        # Prepare test shapes with tuple values.
        # All numbers from 1 to number of columns except 4.
        non_accepted_column_values = tuple(set(range(1, c+1)) - set((4,)))
        tuple_shapes = (
            (random.randint(1, l),
             random.choice(non_accepted_column_values))
            for x in range(N//2))
        # Prepare test shapes with tuple borderline values.
        self.assert_X_from_iterables(
            self.assertRaises,
            itertools.repeat(NotImplementedError),
            itertools.repeat(lambda x: pu.dummy_dataframe(x)),
            tuple_shapes)

        # Test valid shapes with tuple.
        # Prepare test shapes with tuple values.
        borderline_value1 = (1, 4)
        borderline_value2 = (0, 4)
        iterator_of_shapes = (borderline_value1, borderline_value2)
        # Prepare test shapes with tuple borderline values.
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
        pu.statistical_distributions_dataframe((100, 4))

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

        # Test invalid shapes with tuple.
        # Prepare test shapes with tuple values.
        # All numbers from 1 to number of columns except 4.
        tuple_shapes = tuple(
            (random.randint(1, l),
             random.randint(1, c))
            for x in range(N//2))
        # Prepare test shapes with tuple borderline values.
        self.assert_X_from_iterables(
            self.assertEqual,
            tuple_shapes,
            map(lambda x: pu.statistical_distributions_dataframe(x).shape,
                tuple_shapes))

        # Test valid shapes with tuple.
        # Prepare test shapes with tuple values.
        borderline_value1 = (1, 4)
        borderline_value2 = (0, 4)
        iterator_of_shapes = (borderline_value1, borderline_value2)
        # Prepare test shapes with tuple borderline values.
        map_dummy_dataframe = map(lambda x: pu.statistical_distributions_dataframe(x),
                                  iterator_of_shapes)
        self.assert_X_from_iterables(
            self.assertIsInstance,
            map_dummy_dataframe,
            itertools.repeat(pd.DataFrame))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(UtilitiesDataFrames)
    unittest.TextTestRunner(verbosity=2).run(suite)
