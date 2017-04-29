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

from data_utilities import pandas_utilities as pu


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

    This metaclass aims to facilitate iterable assertions (parametrized tests).

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
        def verbosity(cls):
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
                    return cls.verbosity
                # Proceed to one outer frame.
                frame = frame.f_back
            return 0
        # METACLASS: set verbosity.
        classdict['verbosity'] = verbosity
        return type.__new__(mcs, name, bases, classdict)

    def __init__(cls, name, bases, classdict, **kwargs):
        """__init__ method of metaclass."""
        super().__init__(name, bases, classdict)

    def __setattr__(self, name, value):
        """Prevent instances from changing 'non private' attributes.

        If this method is present in the Class level it then prevents instances
        from changing their own attributes.

        """
        try:
            getattr(self, name)
            # May raise error if it is not a private attribute. This allows
            # unittest to change attributes starting with underlines.
            if not name.startswith('_'):
                # If used inside the module test raise no exception.
                if not is_inside_unittest():
                    super().__setattr__(name, value)
                # If trying to set an attribute which already exists raise
                # exception.
                else:
                    raise ValueError(
                        "Cannot change non-private attribute '{0}' of class "
                        "'{1}'.".format(name, self.__class__))
        except AttributeError:
            super().__setattr__(name, value)


class TestDataUtilitiesTestCase(unittest.TestCase, metaclass=TestMetaClass):
    """Test class which will parent all other tests.

    It initialize some class constants and possibly costly operations like
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

    is_inside_unittest = is_inside_unittest()
    N = 50
    N_GRAPHICAL = 3
    l = 100
    c = 10
    SAVE_IMAGES = False
    data = pu.statistical_distributions_dataframe(shape=(l, c))


class TestSupport(TestDataUtilitiesTestCase,
                  metaclass=TestMetaClass):
    """Test class for matplotlib_utilitlies."""

    @classmethod
    def setUpClass(cls):
        """setUp method from unittest.

        Initialize random values to be tested and borderline cases.

        """
        pass

    @classmethod
    def tearDownClass(cls):
        """setUp method from unittest."""
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
        command_string = 'python3 -m unittest -q {0} 2>/dev/null'.format(
            os.path.abspath(__file__))
        command_call = command_string
        return_value = os.system(command_call)
        self.assertEqual(return_value, 0)

    def _test_invoking_test_function(self):
        code_string = ('import data_utilities.tests.test_support as dut; '
                       'dut.is_inside_unittest()')
        command_string = 'python3 -c \'{0}\' 2>/dev/null'
        command_call = command_string.format(code_string)
        return_value = os.system(command_call)
        self.assertEqual(return_value, 0)
