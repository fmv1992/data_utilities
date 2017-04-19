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

from test_support import DataUtilitiesTestCase, TestMetaClass
from data_utilities import matplotlib_utilities as mu


class TestMatplotlibUtilities(DataUtilitiesTestCase, metaclass=TestMetaClass):
    """Test class for matplotlib_utilitlies."""

    @classmethod
    def setUpClass(cls):
        """setUp method from unittest.

        Initialize random values to be tested and borderline cases.

        """
        # Single axes 2d figures (no colorbar or other features).
        cls.figures_2d_histogram = cls.generate_test_figures_2d_histogram()
        # Single axes 3d figures (no colorbar or other features).
        cls.figures_3d = cls.generate_test_figures_3d()

    @classmethod
    def tearDownClass(cls):
        """setUp method from unittest."""
        if cls.SAVE_IMAGES:
            for i, f in enumerate(itertools.chain(cls.figures_2d_histogram,
                                                  cls.figures_3d)):
                f.savefig('/tmp/teardown_{0}.png'.format(i), dpi=300)

    @classmethod
    def generate_test_figures_2d_histogram(cls):
        """Generate a tuple of 2d figures."""
        # Create series.
        iterable_of_series = (pd.Series(np.random.normal(size=cls.l))
                              for _ in range(cls.N_GRAPHICAL))

        # Create figures from series.
        figures = tuple(map(
            cls.figure_from_plot_function,
            itertools.repeat(lambda x: sns.distplot(x, kde=False)),
            iterable_of_series))

        return figures

    @classmethod
    def generate_test_figures_2d(cls):
        """Generate a tuple of 2d figures."""
        pass

    @classmethod
    def generate_test_figures_3d(cls):
        """Generate a tuple of 333figures."""
        # Create random shapes for multi index series.
        # Include these scenarios then fill the rest with random values.
        include_3d_shapes = ((1, 1), (20, 20), (1, 100))
        random_3d_shapes = (
            (random.randint(1, cls.N_GRAPHICAL),
             random.randint(1, cls.N_GRAPHICAL))
            for x in range(cls.N_GRAPHICAL - len(include_3d_shapes)))
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
        """Select each figure before calling plot_3d."""
        fig = plt.figure()
        plot_function(*plot_3d_args)
        return fig

    def setUp(self):
        """setUp method from unittest."""
        self.start_time = time.time()

    def tearDown(self):
        """tearDown method from unittest."""
        elapsed_time = time.time() - self.start_time
        print('\t{0:.3f} seconds elapsed\t'.format(elapsed_time),
              end='',
              flush=True)

    def test_plot_3d(self):
        """plot 3d test."""
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
        """Test label_containers function."""
        map_label_containers = map(
            mu.label_containers,
            (x.axes[0] for x in self.figures_2d_histogram))

        self.assert_X_from_iterables(
            self.assertIsInstance,
            itertools.chain.from_iterable(map_label_containers),
            itertools.repeat(matplotlib.text.Text))
