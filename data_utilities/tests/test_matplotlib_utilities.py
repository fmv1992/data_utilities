"""Test matplolib_utilities from this module."""

import itertools
import random
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_utilities.tests.test_support import (
    TestDataUtilitiesTestCase, TestMetaClass)

from data_utilities import matplotlib_utilities as mu


class TestMatplotlibUtilities(TestDataUtilitiesTestCase,
                              metaclass=TestMetaClass):
    """Test class for matplotlib_utilitlies."""

    @classmethod
    def setUpClass(cls):
        """setUpClass class method from unittest.

        Initialize figure as attributes.

        """
        # TODO: fix the 2 * N//2 != N issue that may happen.

        # Single axes 2d figures (no colorbar or other features).
        cls.figures_2d_histogram = cls.generate_test_figures_2d_histogram()
        # Single axes 3d figures (no colorbar or other features).
        cls.figures_3d = cls.generate_test_figures_3d()

    @classmethod
    def tearDownClass(cls):
        """tearDownClass class method from unittest.

        Save figures in a temporary folder.

        """
        # TODO: save figures in temporary folder.
        if cls.save_images:
            for i, f in enumerate(itertools.chain(cls.figures_2d_histogram,
                                                  cls.figures_3d)):
                f.savefig('/tmp/teardown_{0}.png'.format(i), dpi=300)

    @classmethod
    def generate_test_figures_2d_histogram(cls):
        """generate_test_figures_2d_histogram class method.

        Generate a tuple of 2d histogram figures.

        """
        # Create series.
        iterable_of_series = (pd.Series(np.random.normal(size=cls.n_lines))
                              for _ in range(cls.n_graphical_tests//2))

        # Create figures from series.
        figures = tuple(map(
            cls.figure_from_plot_function,
            itertools.repeat(lambda x: sns.distplot(x, kde=False)),
            iterable_of_series))

        return figures

    @classmethod
    def generate_test_figures_2d(cls):
        """generate_test_figures_2d class method.

        Generate a tuple of 2d figures.

        """
        # TODO: implement or delete method
        pass

    @classmethod
    def generate_test_figures_3d(cls):
        """generate_test_figures_3d class method.

        Generate a tuple of 3d figures.

        """
        # Create random shapes for multi index series.
        # Include these scenarios then fill the rest with random values.
        include_3d_shapes = ((1, 1), (20, 20), (1, 100))
        random_3d_shapes = (
            (random.randint(1, 100),
             random.randint(1, 100))
            for x in range(cls.n_graphical_tests//2 - len(include_3d_shapes)))
        shapes = tuple(itertools.chain(include_3d_shapes, random_3d_shapes))

        # Creates a map of multi index from random shapes tuples.
        # TODO: improve readability.
        map_of_multi_indexes = map(
            lambda x: pd.MultiIndex.from_tuples(tuple(itertools.product(
                range(x[0]),
                (chr(y + ord('a')) for y in range(x[1])))),
                names=('x1', 'x2')),
            shapes)

        # Create series from multi indexes.
        map_of_series = map(lambda x: pd.Series(np.random.normal(size=len(x)),
                                                index=x),
                            map_of_multi_indexes)

        # Create figures from series.
        figures = tuple(map(cls.figure_from_plot_function,
                            itertools.repeat(mu.plot_3d),
                            map_of_series))

        return figures

    @classmethod
    def figure_from_plot_function(cls, plot_function, *plot_3d_args, **kwargs):
        """Initialize a figure and call a plot function on it."""
        fig = plt.figure()
        plot_function(*plot_3d_args)
        return fig

    def setUp(self):
        """setUp method from unittest.

        Store start time for test method. Running time is computed and
        displayed in the tearDown method.

        """
        self.start_time = time.time()

    def tearDown(self):
        """tearDown method from unittest.

        Compute and display running time for test method.

        """
        elapsed_time = time.time() - self.start_time
        if self.verbose:
            print('\t{0:.3f} seconds elapsed\t'.format(elapsed_time), end='',
                  flush=True)

    def test_plot_3d(self):
        """Plot 3d test."""
        # Test 3d plots.
        self.assert_X_from_iterables(
            self.assertIsInstance,
            # TODO: error prone since colorbars can be added.
            (fig.get_axes()[0] for fig in self.figures_3d),
            itertools.repeat(Axes3D))
        # Test that there is just one axes per figure.
        for i, figure in enumerate(self.figures_3d):
            axes = figure.get_axes()
            if len(axes) != 1:
                # TODO: colorbar may add a second axes.
                pass
                raise ValueError(
                   "Axes has the wrong number of elements: {0} but "
                   "should be 1.".format(len(axes)))

    def test_label_containers(self):
        """Label containers test."""
        map_label_containers = map(
            mu.label_containers,
            (x.axes[0] for x in self.figures_2d_histogram))

        self.assert_X_from_iterables(
            self.assertIsInstance,
            itertools.chain.from_iterable(map_label_containers),
            itertools.repeat(matplotlib.text.Text))
