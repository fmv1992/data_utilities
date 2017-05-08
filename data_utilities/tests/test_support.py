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


import unittest
import inspect
import os
import tempfile


from data_utilities import pandas_utilities as pu
import numpy as np


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

        @property
        def verbose(cls):
            """Return the verbosity setting of the currently running unittest.

            This function 'scans' the frame looking for a 'cls' variable.

            Returns:
                int: the verbosity level.

                0 if this is the __main__ file
                1 if run with unittests module without verbosity (default in
                TestProgram)
                2 if run with unittests module with verbosity
            """
            frame = inspect.currentframe()
            # Scans frames from innermost to outermost for a TestProgram
            # instance.  This python object has a verbosity defined in it.
            while frame:
                cls = frame.f_locals.get('cls')
                if isinstance(cls, unittest.TestProgram):
                    return cls.verbose
                # Proceed to one outer frame.
                frame = frame.f_back
            return 0
        # METACLASS: set verbosity.
        classdict['verbose'] = verbose
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
        if hasattr(self.__class__, name):
            raise AttributeError(
                ("Cannot change non-private attribute '{0}' of class '{1}'."
                 ).format(name, self.__class__))
        else:
            super().__setattr__(name, value)

    @classmethod
    def compose_functions(cls,
                          x,
                          number_of_compositions=1,
                          functions=(np.sin, np.exp, np.square, np.polyval,
                                     np.tan, ),
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
        i = 0
        y = x
        while i < number_of_compositions:
            func = np.random.choice(functions)
            if func == np.polyval:
                n_coefs = np.random.randint(0, 10)
                coefs = np.random.randint(-50, 50, size=n_coefs)
                y = func(coefs, x)
            else:
                y = func(y)
            if clip_big_values:
                y = np.clip(y, -clip_value, clip_value)
            i += 1
        return y

    # Goes with 'fast' parameters by default.
    is_inside_unittest = is_inside_unittest()
    n_tests = 5
    n_graphical_tests = 3
    n_lines = 50
    n_columns = 5
    save_figures = False
    maxDiff = None  # TODO: check if it is being utilized
    data = pu.statistical_distributions_dataframe(shape=(n_lines, n_columns))

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

    def test_test_framework(self):
        """Test is_inside_unittest function."""
        if is_inside_unittest():
            self._test_invoking_test_function()
        else:
            self._test_invoking_unittest()

    def _test_invoking_unittest(self):
        """Private method which invokes a python unittest separate process."""
        command_string = 'python3 -m unittest -q {0} 2>/dev/null'.format(
            os.path.abspath(__file__))
        command_call = command_string
        return_value = os.system(command_call)
        self.assertEqual(return_value, 0)

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
