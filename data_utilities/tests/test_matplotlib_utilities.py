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


def setUpModule():
    """Setup TestDataUtilitiesTestCase 'data' attribute.

    Useful if there is a unittest being run.
    """
    TestDataUtilitiesTestCase.update_data()


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
        cls.figures_2d_histogram = cls._generate_test_figures_2d_histogram()
        # Single axes 3d figures (no colorbar or other features).
        ## TODO change `figures_3d` name to bar3d or something
        cls.figures_3d = cls._generate_bar3d_test_figures()
        ## TODO change `figures_3d` name to bar3d or something
        cls.figures_3d_scatterplot = cls._generate_scatterplot_test_figures()
        # Single axes 3d scatter plt.
        # figures
        # ├── figures_2d
        # │   ├── histogram
        # │   ├── lines
        # │   └── scatter
        # └── figures_3d
        #     ├── bars
        #     └── scatter
        # TODO: What does matplotlib utilities need?
        ##  an all figures attribute.
        cls.figures = None
        # TODO: What does matplotlib utilities need?
        ##  an attribute with 2d figures.
        cls.figures_2d = None
        cls.figures_2d_histogram = None
        cls.figures_2d_lines = None
        cls.figures_2d_scatter = None
        # TODO: What does matplotlib utilities need?
        ##  an attribute with 3d figures.
        cls.figures_3d = None
        cls.figures_3d_bars = None
        cls.figures_3d_scatter = None

    @classmethod
    def tearDownClass(cls):
        """tearDownClass class method from unittest.

        Save figures in a temporary folder.

        """
        # TODO: save figures in temporary folder.
        if cls.save_figures:
            for i, f in enumerate(itertools.chain(cls.figures_2d_histogram,
                                                  cls.figures_3d)):
                f.savefig('/tmp/teardown_{0}.png'.format(i), dpi=300)

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

    @classmethod
    def _generate_test_figures_2d_histogram(cls):
        """_generate_test_figures_2d_histogram class method.

        Generate a tuple of 2d histogram figures.

        """
        # Create series. Will be divided by more than //2 when all plots are
        # ready.
        def dist_function01(): return np.random.normal(size=cls.n_lines)

        def dist_function02(): return np.random.randint(
            0,
            99999) * np.arange(cls.n_lines)

        def dist_function03(): return np.random.randint(
            0,
            99999) * np.ones(cls.n_lines)
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
    def _generate_test_figures_2d(cls):
        """_generate_test_figures_2d class method.

        Generate a tuple of 2d figures.

        """
        # TODO: implement or delete method
        pass

    @classmethod
    def _generate_bar3d_test_figures(cls):
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
    def _generate_xyz_meshgrid(cls):
        """Generate a tuple of (x, y, z) meshgrids.

        Generate a tuple of meshgrids. Intended to help
        _generate_scatterplot_test_figures.

        """
        n_points = 100
        x_start = random.randint(0, 1000)
        x_end = x_start + random.randint(1, 1000)
        x = np.linspace(x_start, x_end, n_points)
        y = x / random.randint(1, 100)

        xx, yy = np.meshgrid(x, y)

        zz = np.random.choice((np.sum, np.multiply))(xx, yy)

        zz = cls.compose_functions(zz)

        return (xx, yy, zz)

    @classmethod
    def _generate_scatterplot_test_figures(cls):
        """Generate scatter plot test figures class method.

        Generate a tuple of 3d figures.

        """
        # Test methodology: Create test support.

        # Test methodology: Create test cases.
        #   Create a test variable from `test = self.create_test_cases()`
        test = cls.create_test_cases(
            cls._generate_xyz_meshgrid,
            is_graphical_test=True)

        # Test methodology: Create test objects from test cases.
        #   Create test figures, texts, tuples, etc from test cases.
        test_figures = map(
            cls.figure_from_plot_function,
            itertools.repeat(plt.scatter),
            test,
            itertools.repeat({'projection': '3d'})
        )
        pass


        # Test methodology: Run tests.
        #   Has to invoke `self.assert_X_from_iterables()`

        # Test methodology: Save persistency.
        #   (Optional) Save figures as needed
        # if self.save_figures or True:
        #     for i, f in enumerate(test_figures):
        #         f.savefig(
        #             '/tmp/test_add_summary_statistics_textbox_{0}.png'.format(
        #                 i),
        #             dpi=300)
        #######################################################################
        ##########################   OLD   ####################################
        #######################################################################
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
        #######################################################################
        #######################################################################
        #######################################################################

        return figures

    @classmethod
    def figure_from_plot_function(cls, plot_function, *plot_3d_args, **kwargs):
        """Initialize a figure and call a plot function on it."""
        fig = plt.figure()
        plot_function(*plot_3d_args, **kwargs)
        return fig

    def test_add_summary_statistics_textbox(self):
        """Add summary statistics textbox test."""
        # Test methodology: Create test support.
        # Initialize x values.
        x = np.linspace(-10, 10, self.n_lines)
        # Initilize a random function to generate tests.
        generate_random_test_function = lambda: self.compose_functions(x, 3)

        # Test methodology: Create test cases.
        test = self.create_test_cases(
            generate_random_test_function,
            np.ones(self.n_lines),  # borderline
            np.zeros(self.n_lines),  # borderline
            container_function=pd.Series)
        test = tuple(test)

        # Test methodology: Create test figures from test cases.
        test_figures = tuple(map(
            lambda z: self.figure_from_plot_function(plt.plot, x, z), test))

        # Test methodology: Create test texts from test cases.
        test_texts = map(
            mu.add_summary_statistics_textbox,
            test,
            (x.gca() for x in test_figures))
        test_texts = tuple(test_texts)
        # Check texts.
        if not test_texts:
            raise Exception("Texts were not created accordingly.")

        # Test methodology: Run tests.
        # Test this function with:
        # Scatter plot.     # Done.
        # Bar plot.         # TODO
        self.assert_X_from_iterables(
            self.assertIsInstance,
            test_texts,
            itertools.repeat(matplotlib.text.Text))

        # TODO: remove this short circ.
        if self.save_figures or True:
            for i, f in enumerate(test_figures):
                f.savefig(
                    '/tmp/test_add_summary_statistics_textbox_{0}.png'.format(
                        i),
                    dpi=300)

    def test_histogram_of_categorical(self):
        """Histogram of categorical test."""
        # TODO: implement.
        pass
    def test_histogram_of_dataframe(self):
        """Histogram of dataframe test."""
        # TODO: Comply with `test_structure_for_mu_tests` structure.
        tuple_of_figures_of_histograms = mu.histogram_of_dataframe(self.data)

        self.assertTrue(isinstance(tuple_of_figures_of_histograms, tuple))

        self.assert_X_from_iterables(
            self.assertIsInstance,
            tuple_of_figures_of_histograms,
            itertools.repeat(plt.Figure))

        if self.save_figures:
            for i, f in enumerate(tuple_of_figures_of_histograms):
                f.savefig('/tmp/test_histogram_of_dataframe_{0}.png'.format(i),
                          dpi=300)

    def test_histogram_of_floats(self):
        """Histogram of floats test."""
        # TODO: implement.
        pass
    def test_histogram_of_integers(self):
        """Histogram of integers test.

        Test non contiguous blocks of data. The input for the test is a group
        of non contiguous ranges.

        These ranges are mapped with figures to a function that returns figures
        with histograms on it.

        """
        # Test methodology: Create test support.
        generate_random_test_function = lambda: tuple(itertools.chain(
            range(
                random.randint(0, 5),
                random.randint(10, 15)),  # Has a maximum of 15 elements.
            range(
                random.randint(10000, 10000+5),
                random.randint(10000+10, 10000+15))))   # Has a maximum of
                                                        # 15 elements.

        # Test methodology: Create test cases.
        borderline_range01 = tuple(itertools.chain(
            range(-10000, -9995),
            range(-10, 10),
            range(100, 110),
            range(10000, 10010)))
        borderline_range02 = tuple(itertools.chain(
            range(-500, -490),
            range(500, 510),
            range(1000, 1010)))
        borderline_range03 = tuple(itertools.chain(
            range(10),
            range(100000, 100005)))
        test = self.create_test_cases(
            generate_random_test_function,
            borderline_range01,
            borderline_range02,
            borderline_range03,
            is_graphical_test=True)

        # Test methodology: Create test figures from test cases.
        def dist_plot_no_kde(fig, a):
            ax = fig.gca()
            mu.histogram_of_integers(a, kde=False, ax=ax)
            return fig
        test_figures_histogram = tuple(map(
            dist_plot_no_kde,
            itertools.repeat(plt.figure()),
            test))

        # Test methodology: Run tests.
        self.assert_X_from_iterables(
            self.assertIsInstance,
            (x.gca() for x in test_figures_histogram),
            itertools.repeat(matplotlib.axes.Axes))

        # Test methodology: Save persistency.
        if self.save_figures:
            for i, f in enumerate(test_figures_histogram):
                f.savefig('/tmp/test_histogram_of_integers_{0}.png'.format(i),
                          dpi=300)

    def test_label_containers(self):
        """Label containers test."""
        # TODO: Comply with `test_structure_for_mu_tests` structure.
        map_label_containers = map(
            mu.label_containers,
            (x.axes[0] for x in self.figures_2d_histogram))

        self.assert_X_from_iterables(
            self.assertIsInstance,
            itertools.chain.from_iterable(map_label_containers),
            itertools.repeat(matplotlib.text.Text))

    def test_list_usable_backends(self):
        """List usable backends test."""
        self.assertTrue(isinstance(mu.list_usable_backends(), list))

    def test_plot_3d(self):
        """Plot 3d test."""
        # TODO: Comply with `test_structure_for_mu_tests` structure.
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

    def test_scale_axes_axis(self):
        """Scale axes axis test."""
        # Test methodology: Create test support.
        #   Create support variables for tests.

        # Test methodology: Create test cases.
        #   Create a test variable from `test = self.create_test_cases()`

        # Test methodology: Create test objects from test cases.
        #   Create test figures, texts, tuples, etc from test cases.

        # Test methodology: Run tests.
        #   Has to invoke `self.assert_X_from_iterables()`

        # Test methodology: Save persistency.
        #   (Optional) Save figures as needed
        # if self.save_figures or True:
        #     for i, f in enumerate(test_figures):
        #         f.savefig(
        #             '/tmp/test_add_summary_statistics_textbox_{0}.png'.format(
        #                 i),
        #             dpi=300)
        # TODO: implement.
        pass
        # Test this attribute
        self.figures_3d_scatterplot

