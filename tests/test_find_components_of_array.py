"""Test pandas_utilities from this module."""


# import pprint
import unittest
import pandas_utilities as pu
import numpy as np
from test_metaclass import TestMetaClass

N = 1000
l = 100
c = 15


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
