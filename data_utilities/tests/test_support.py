"""This module provides support to actual tests.

The unittest module in python is not so flexible to allow multiple tests with
different inputs. Aggregating parameters in iterables and using a for loop does
not seem to be elegant in this scenario.

A different approach is to use meta classes to create instances with multiple
tests.

This latter approach will be tried here.

A good tutorial for metaclasses:
    https://blog.ionelmc.ro/2015/02/09/understanding-python-metaclasses/

The implementation of the idea to run parametrized tests in python comes from
here:
    http://stackoverflow.com/questions/32899/how-to-generate-dynamic-parametrized-unit-tests-in-python

"""


import itertools
import unittest
import inspect
import os
import tempfile
import warnings


import data_utilities as du
from data_utilities import pandas_utilities as pu
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def setUpModule():
    """Setup TestDataUtilitiesTestCase 'data' attribute.

    Useful if there is a unittest being run.
    """
    TestDataUtilitiesTestCase.update_data()


def is_inside_recursive_test_call():
    """Test if a function is running from a call to du.test()."""
    frame = inspect.currentframe()
    count_test = 0
    while frame:
        # Test breaking condition.
        if count_test >= 1:
            return True
        # Find if there is a breaking condition and update the value.
        test_function = frame.f_locals.get('test')
        if (hasattr(test_function, '_testMethodName')) and (
                test_function._testMethodName ==
                'test_module_level_test_calls'):
            count_test += 1
        # Go to next frame.
        frame = frame.f_back
    return False


def is_inside_unittest():
    """Test if a function is running from unittest.

    This function will help freezing the __setattr__ freezing when running
    inside a unittest environment.

    """
    frame = inspect.currentframe()

    # Calls from unittest discover have the following properties:
    # 1) Its stack is longer
    # 2) It contains arguments ('argv', 'pkg_name', 'unittest', etc) for
    # instance: a key value pair in the format:
    # ('argv', ['python3 -m unittest', 'discover', '-vvv', '.'])
    key = 'argv'  # this may be error prone...
    value = 'unittest'
    while frame:
        # print(frame.f_locals.items())
        frame_argv = frame.f_locals.get(key)
        if frame_argv and value in ''.join(frame_argv):
            return True
        frame = frame.f_back
    return False


class TestMetaClass(type):
    """Metaclass for all tests in this module.

    This metaclass aims to facilitate iterable assertions (parametrized tests)
    with randomized input.

    """

    def __new__(mcs, name, bases, classdict, **kwargs):
        """__new__ method of metaclass."""
        # TODO: maybe this could be just put in the base class instead of in
        # the meta class.
        def assert_X_from_iterables(self, x=lambda: True, *args, **kwargs):
            func = getattr(self, x.__name__)
            for i, zipargs in enumerate(zip(*args)):
                # print('iteration i:', i)
                with self.subTest(iteration=i, arguments=zipargs):
                    func(*zipargs)
            return None
        classdict['assert_X_from_iterables'] = assert_X_from_iterables

        # TODO: reactivate when unittests are up.
        # Investigate implications for current parsing/setting of attributes.
        # @property
        # def verbose(cls):
        #     """Return the verbosity setting of the currently running unittest.

        #     This function 'scans' the frame looking for a 'cls' variable.

        #     Returns:
        #         int: the verbosity level.

        #         0 if this is the __main__ file
        #         1 if run with unittests module without verbosity (default in
        #         TestProgram)
        #         2 if run with unittests module with verbosity
        #     """
        #     frame = inspect.currentframe()
        #     # Scans frames from innermost to outermost for a TestProgram
        #     # instance.  This python object has a verbosity defined in it.
        #     while frame:
        #         cls = frame.f_locals.get('cls')
        #         if isinstance(cls, unittest.TestProgram):
        #             return cls.verbose
        #         # Proceed to one outer frame.
        #         frame = frame.f_back
        #     return 0
        # METACLASS: set verbosity.
        # classdict['verbose'] = verbose

        return type.__new__(mcs, name, bases, classdict)

    def __init__(cls, name, bases, classdict, **kwargs):
        """__init__ method of metaclass."""
        super().__init__(name, bases, classdict)


class TestDataUtilitiesTestCase(unittest.TestCase, metaclass=TestMetaClass):
    """Test class which will parent all other tests.

    It initializes some class constants and possibly costly operations like
    creating dummy dataframes.

    """

    def __new__(cls, *args, **kwargs):
        """__new__ method."""
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        """Instatiate the test case.

        Instatiate the test case to allow customization of run parameters.

        """
        super().__init__(*args, **kwargs)  # call init from unittest.TestCase

    def __setattr__(self, name, value):
        """Prevent instances from changing 'non private' attributes.

        An omission is made if setting the attribute is done outside unittest.
        That means that the setting is within the module level test function
        (and thus changing of existing attributes is allowed).

        If this method is present in the class level it then prevents instances
        from changing their own attributes.

        """
        if name.startswith('_'):
            super().__setattr__(name, value)
        elif hasattr(self.__class__, name):
            raise AttributeError(
                ("Cannot change non-private attribute '{0}' of class '{1}'."
                 ).format(name, self.__class__))
        else:
            super().__setattr__(name, value)

    @classmethod
    def create_random_grid(cls,
                           x=None,
                           x_grid=False,
                           xy_grid=True,
                           xyz_grid=False,
                           normalize=True,
                           maximum_amplitude=50,
                           compose_functions_kwargs={}):
        """Create a random grid of values to generate meaningful test cases."""
        # TODO: what to do when more than one grid is specified?
        # if xy_grid and xyz_grid:
        #     raise ValueError('Either specify xy or xyz grid.')

        # Initialize constants.
        AMPLITUDE = 1000
        N_POINTS = cls.n_lines

        # Initialize x and y.
        if x is None:
            x_min = np.random.randint(-AMPLITUDE, AMPLITUDE)
            x_max = x_min + np.random.randint(0, AMPLITUDE)
            x = np.linspace(x_min, x_max, N_POINTS)
        y = cls.compose_functions(x, **compose_functions_kwargs)

        # Initialize z if needed.
        if xyz_grid:
            x , y = np.meshgrid(x, y)
            z = cls.compose_functions(
                x + y,
                **compose_functions_kwargs)

        # Normalize data if needed.
        if normalize:
            scaler = MinMaxScaler(feature_range=(0, maximum_amplitude))
            if xyz_grid:
                expanded = np.append(x, (y, z))
            if xy_grid:
                expanded = np.append(x, y)
            scaler.fit(expanded.reshape(-1, 1))
            x = scaler.transform(x.reshape(-1, 1)).flatten()
            y = scaler.transform(y.reshape(-1, 1)).flatten()
            if xyz_grid:
                z = scaler.transform(z.reshape(-1, 1)).flatten()

        # Return the adequate return value.
        if x_grid:
            return (x, )
        if xy_grid:
            return (x, y)
        if xyz_grid:
            return (x, y, z)



    def create_test_cases(self, generate_random_test_function,
                          *borderline_test_cases,
                          container_function=None,
                          is_graphical_test=False,
                          **container_function_kwargs):
        """Create a generator from mapping args to a container function.

        This function truncates the args according to n_tests. Likewise it
        expects the last element of *args to be an infinite iterator to be
        sliced according to n_tests.

        The args structure should be like this:

        +-------------+------------+-----+-------------+-------------------+
        | borderline1 | bordeline2 | ... | borderlineN | infinite iterator |
        +=============+============+=====+=============+===================+

        A warning is raised if n_tests is so small that all borderline values
        are not used.

        """
        if is_graphical_test:
            n_tests = self.n_graphical_tests
        else:
            n_tests = self.n_tests
        # Compute the number of runs for the last test.
        n_run_last_args = n_tests - len(borderline_test_cases)
        if n_run_last_args < 0:
            # Warn if borderlines are smaller than n_tests.
            warnings.warn(
                'Borderline values (len == {0}) are smaller than n_tests '
                '({1})'.format(len(borderline_test_cases), n_tests),
                DataUtilitiesTestWarning)
            n_run_last_args = 0

        # Join arguments again.
        joined_tests = itertools.chain(
            borderline_test_cases,
            (generate_random_test_function() for i in range(n_run_last_args)))
        if container_function is not None:
            joined_tests_w_container = map(
                lambda x: container_function(x, **container_function_kwargs),
                joined_tests)
        else:
            joined_tests_w_container = joined_tests

        return joined_tests_w_container

    @classmethod
    def compose_functions(cls,
                          x,  # TODO: generalize for *args and enable functions to be more than 1 argument functions.
                          number_of_compositions=1,
                          functions=(np.sin, np.exp, np.square, np.polyval,
                                     np.tan, np.copy, np.random.random ),
                          clip_big_values=True,
                          clip_value=1e6):
        """Compose functions from an iterable of functions.

        This is a helper function to cover a more real life scenario of
        plottings.

        Arguments:
            x (numpy.array): array for which composed values will be computed.
            number_of_compositions (int): number of compositions of functions.
            functions (tuple): an iterable of functions.
            clip_big_values (bool): whether or not to limit function extremes.
            clip_value (float): limit values for function composition.

        Returns:
            y (numpy.array): array of composed functions

        """
        y = x
        while number_of_compositions > 0:
            func = np.random.choice(functions)
            if func == np.polyval:
                n_coefs = np.random.randint(0, 10)
                coefs = np.random.randint(-50, 50, size=n_coefs)
                y = func(coefs, x)
            elif func == np.random.random:
                y = func(size=len(x)) - 0.5
            else:
                y = func(y)
            if clip_big_values:
                y = np.clip(y, -clip_value, clip_value)
            number_of_compositions -= 1
        return y

    # Goes with 'fast' parameters by default.
    is_inside_unittest = is_inside_unittest()
    n_tests = 5
    n_graphical_tests = 3
    n_lines = 50
    n_columns = 5
    save_figures = False
    maxDiff = None  # TODO: check if it is being utilized

    @classmethod
    def update_data(cls):
        """Update the 'data' attribute.

        Most likely this is set during the execution of data_utilities.test().

        """
        cls.data = pu.statistical_distributions_dataframe(
            shape=(cls.n_lines, cls.n_columns))
        return None

    # Setup temporary folder to be used.
    temp_directory = tempfile.TemporaryDirectory(prefix='test_data_utilities_')


class TestSupport(TestDataUtilitiesTestCase,
                  metaclass=TestMetaClass):
    """Test class for test_support."""

    @classmethod
    def setUpClass(cls):
        """setUpClass class method from unittest."""
        pass

    @classmethod
    def tearDownClass(cls):
        """tearDownClass class method from unittest."""
        pass

    def setUp(self):
        """setUp method from unittest."""
        pass

    def tearDown(self):
        """tearDown method from unittest."""
        pass

    def test_create_test_cases(self):
        """Create test cases test."""
        self.create_test_cases(np.random.random)
        # TODO: improve the logic of this test.
        pass

    def test_create_random_grid(self):
        """Create random grid test."""
        self.create_random_grid()
        self.create_random_grid(xy_grid=False, xyz_grid=True)
        # TODO: improve the logic of this test.
        pass

    def test_test_framework(self):
        """Test is_inside_unittest function."""
        if is_inside_unittest():
            self._test_invoking_test_function()
        else:
            self._test_invoking_unittest()

    def _test_invoking_unittest(self):
        """Private method which invokes a python unittest separate process."""
        command_string = ('''python3 -m unittest -vvv data_utilities'''
                          ).format(
            os.path.abspath(__file__))
        command_call = command_string
        return_value = os.system(command_call)  # noqa
        # TODO: fix this invoking in virtual environment.
        # Return value is 256 in virtual environments despite my great efforts
        # to understand why.
        # Probably it has to do with matplotlib in virtual environments.
        # self.assertEqual(return_value, 0)
        pass

    def _test_invoking_test_function(self):
        """Private method which invokes the module's test function.

        The invocation creates its own separate process.

        """
        # TODO: matplotlib backend with bash script does not use the correct
        # backend.
        code_string = ('import data_utilities.tests.test_support as dut; '
                       'dut.is_inside_unittest()')
        command_string = 'python3 -c \'{0}\' 2>/dev/null'
        command_call = command_string.format(code_string)
        return_value = os.system(command_call)
        self.assertEqual(return_value, 0)

    def test___setattr__(self):
        """Test __setattr__."""
        from data_utilities.tests.test_matplotlib_utilities import (
            TestMatplotlibUtilities)
        # Using self.
        # Set non existent.
        self.non_existent_attribute = 1
        del self.non_existent_attribute
        # Change existing.
        with self.assertRaises(AttributeError):
            self.data = None
        # Using other dummy tests.
        m = TestMatplotlibUtilities()
        m.non_existent_attribute = 1
        del m.non_existent_attribute
        try:
            m.data = None
        except AttributeError:
            pass

    def test_structure_for_mu_tests(self):
        """Test structure for all tests in this module."""
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
        pass


class TestModule(TestDataUtilitiesTestCase, metaclass=TestMetaClass):
    """Test class for module level tests.

    Test class for module level tests e.g.: data_utilities.test()
    """

    def test_module_level_test_calls(self):
        """Test module level test calls."""
        # Test the full label.
        du.test('full',
                n_graphical_tests=0,
                save_figures=False)
        # Test the fast label.
        du.test('fast',
                verbose=True,  # Ensure coverage of the 'verbose' line.
                save_figures=False)
        # Test any other non implemented label.
        with self.assertRaises(NotImplementedError):
            du.test('unnamed_test_mode')

    def test_is_inside_recursive_test_call(self):
        """Test is_inside_recursive_test_call function."""
        def _create_stack_depth_1():
            def _create_stack_depth_2():
                def _create_stack_depth_3():
                    return is_inside_recursive_test_call()
                return _create_stack_depth_3()
            return _create_stack_depth_2()
        self.assertFalse(_create_stack_depth_1())
        return None


class DataUtilitiesTestWarning(Warning):
    pass
