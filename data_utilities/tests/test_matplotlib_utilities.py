"""Test matplolib_utilities from this module."""

import itertools
import os
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from data_utilities import matplotlib_utilities as mu
from data_utilities.tests.test_support import (
    TestDataUtilitiesTestCase, TestMetaClass)


class TestMatplotlibUtilities(TestDataUtilitiesTestCase,
                              metaclass=TestMetaClass):
    """Test class for matplotlib_utilitlies."""

    @classmethod
    def setUpClass(cls):
        """Set up class method from unittest.

        Initialize figure as attributes.

        """
        super().setUpClass()

        # TODO: fix the 2 * N//2 != N issue that may happen.
        # Single axes 2d figures (no colorbar or other features).
        cls.figures_2d_histogram = cls.generate_test_figures_2d_histogram()
        # Single axes 3d figures (no colorbar or other features).
        cls.figures_3d = cls.generate_bar3d_test_figures()

    @classmethod
    def tearDownClass(cls):
        """Tear down class method from unittest.

        Save figures in a temporary folder.

        """
        # It is more memory efficient if every method saves its own figures.
        pass

    @classmethod
    def generate_test_figures_2d_histogram(cls):
        """generate_test_figures_2d_histogram class method.

        Generate a tuple of 2d histogram figures.

        """
        # Create series. Will be divided by more than //2 when all plots are
        # ready.
        def dist_function01(): return np.random.normal(
            size=cls.n_lines_test_pandas)

        def dist_function02(): return np.random.randint(
            0,
            99999) * np.arange(cls.n_lines_test_pandas)

        def dist_function03(): return np.random.randint(
            0,
            99999) * np.ones(cls.n_lines_test_pandas)
        dist_functions = (dist_function01, dist_function02, dist_function03)
        iterable_of_series = (pd.Series(np.random.choice(dist_functions)())
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
    def generate_bar3d_test_figures(cls):
        """Generate bar3d test figures class method.

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
        plot_function(*plot_3d_args, **kwargs)
        return fig

    def setUp(self):
        """Set up method from unittest.

        Store start time for test method. Running time is computed and
        displayed in the tearDown method.

        """
        self.start_time = time.time()

    def tearDown(self):
        """Tear down method from unittest.

        Compute and display running time for test method.

        """
        elapsed_time = time.time() - self.start_time
        if self.verbose:
            print('\t{0:.3f} seconds elapsed\t'.format(elapsed_time), end='',
                  flush=True)
        plt.close('all')

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

    def test_histogram_of_integers(self):
        """Histogram of integers test.

        Test non contiguous blocks of data.

        """
        # Test some borderline cases for histogram of integers.
        borderline_integer_sequence = (
            itertools.chain(range(-10000, -9995),
                            range(-10, 10),
                            range(100, 110),
                            range(10000, 10010)),
            itertools.chain(range(-500, -490),
                            range(500, 510),
                            range(1000, 1010),),
            itertools.chain(range(10),
                            range(100000, 100005),),
        )
        borderline_integer_sequence = tuple(map(tuple,
                                                borderline_integer_sequence))

        # Initialize auxiliar functions.
        def dist_plot_no_kde(fig, a):
            ax = fig.gca()
            mu.histogram_of_integers(a, kde=False, ax=ax)
            return fig
        # Initialize figures.
        figures = map(lambda x: plt.figure(),
                      range(len(borderline_integer_sequence)))
        # Plot histogram on figures.
        figures_histogram = tuple(map(
            dist_plot_no_kde,
            figures,
            borderline_integer_sequence))

        self.assert_X_from_iterables(
            self.assertIsInstance,
            (x.gca() for x in figures_histogram),
            itertools.repeat(matplotlib.axes.Axes))

        if self.save_figures:
            for i, f in enumerate(figures_histogram):
                f.savefig(os.path.join(
                    self.test_directory.name,
                    'test_histogram_of_integers_{0}'.format(i)),
                          dpi=300)

    def test_add_summary_statistics_textbox(self):
        """Add summary statistics textbox test."""
        # Initialize x values.
        x = np.linspace(-10, 10, self.n_lines_test_pandas)

        # Initialize y values.
        # Add some borderline cases to y.
        y_borderline = (pd.Series(np.ones(self.n_lines_test_pandas)),
                        pd.Series(np.zeros(self.n_lines_test_pandas)))
        # Add other functions to y.
        y = map(lambda x, y: pd.Series(self.compose_functions(x, 3)),
                itertools.repeat(x),
                range(self.n_graphical_tests - len(y_borderline)))
        # Join both iterables. Needed in figures and texts.
        y = tuple(itertools.chain(y_borderline, y))

        # Need to implement its own figures factory because it has to give
        # series as argument.
        figures = tuple(map(
            lambda z: self.figure_from_plot_function(plt.plot, x, z), y))

        texts = map(
            mu.add_summary_statistics_textbox,
            y,
            (x.gca() for x in figures))

        texts = tuple(texts)
        if not texts:
            raise Exception("Texts were not created accordingly.")

        # Test this function with:
        # Scatter plot.
        # Bar plot.  # TODO
        self.assert_X_from_iterables(
            self.assertIsInstance,
            texts,
            itertools.repeat(matplotlib.text.Text))

        if self.save_figures:
            for i, f in enumerate(figures):
                f.savefig(os.path.join(
                    self.test_directory.name,
                    'test_add_summary_statistics_textbox_{0}.png'.format(i)),
                          dpi=300)
