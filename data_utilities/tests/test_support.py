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

from data_utilities import pandas_utilities as pu

N = 50
N_GRAPHICAL = 10
l = 1000
c = 20
SAVE_IMAGES = True


class TestMetaClass(type):
    """Metaclass for all tests in this module.

    This metaclass aims to facilitate iterable assertions (parametrized tests).

    """

    def __new__(mcs, name, bases, classdict):
        """__new__ method of metaclass."""
        def assert_X_from_iterables(self, x=lambda: True, *args, **kwargs):
            func = getattr(self, x.__name__)
            for i, zipargs in enumerate(zip(*args)):
                # print('iteration i:', i)
                with self.subTest(iteration=i, arguments=zipargs):
                    func(*zipargs)
            return None
        # METACLASS: create helper function.
        classdict['assert_X_from_iterables'] = assert_X_from_iterables

        # TODO: implement the verbosity attribute.
        @property
        def verbosity(cls):
            """Return the verbosity setting of the currently running unittest.

            This function 'scans' the

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

    def __setattr__(self, name, value):
        """Prevent instances from changing 'non private' attributes.

        If this method is present in the MetaClass level it then prevents
        instances of changing attributes of the class (shared among all
        instances).

        If this method is present in the Class level it then prevents instances
        from changing their own attributes.

        """
        # Disallow the overwritting of attributes.
        try:
            getattr(self, name)
            # Raise error if it is not a private attribute.
            if not name.startswith('_'):
                raise ValueError(
                    "Cannot change non-private attribute '{0}' of class "
                    "'{1}'.".format(name, self.__class__))
        except AttributeError:
            super().__setattr__(name, value)


class DataUtilitiesTestCase(unittest.TestCase, metaclass=TestMetaClass):
    """Test class which will parent all other tests.

    It initialize some class constants and possibly costly operations like
    creating dummy dataframes.

    """

    c = c
    l = l
    N = N
    N_GRAPHICAL = 10
    SAVE_IMAGES = SAVE_IMAGES
    data = pu.statistical_distributions_dataframe(
            shape=(l, c))
