"""Data analysis helper module written in python.

The module relies upon python's scientific/numeric stack to make some
procedures easier.

It offers one module per project (pandas, matplotlib, python itself). Each
module should mimick the development guidelines of its parent module.

"""
# According to this, a version number is defined by
#
# MAJOR.MINOR.PATCH
#
# where
#
# MAJOR version when you make incompatible API changes
# MINOR version when you add functionality in a backwards-compatible manner,
# and
# PATCH version when you make backwards-compatible bug fixes.
#
# According to this and this Python 3.5.0 was released in 2015-09-13, while
# Python 3.4.0 was released on March 16th, 2014.
#
# The third number in the version number is the PATCH which usually fixes bugs,
# so the last version of Python is 3.6.0 which has no patches so far. I
# recommend to use the version based on the compatibility of the libraries you
# are going to use.

# User version numbering from here.
# http://stackoverflow.com/questions/42259098/python-version-numbering-scheme/42259144

import unittest

import data_utilities.tests.test_pandas_utilities as tpu
import data_utilities.tests.test_matplotlib_utilities as tmu
import data_utilities.tests.test_python_utilities as tpyu
import data_utilities.tests.test_support as ts
from data_utilities.tests.test_support import TestDataUtilitiesTestCase

__version__ = '1.2.7'


def test(label='full', verbose=1, N=50, lines=100, columns=10, N_GRAPHICAL=3):
    """Module level test function.

    Run tests using the unittest module. Both 'numpy style' and unittest
    invocations should work:
    `python3 -m unittest discover -vvv data_utilities/tests`
    `python3 -c "import data_utilities as du; du.test()"`

    Created based on the same architecture as the scipy test function defined
    on their __init__.py.

    Test are modularized and their repetition is customizable.

    Arguments:
        label (str): test suite to be implemented. Either 'full' or 'fast'. Not
        implemented.
        verbose (int): verbosity of the test suite. Not implemented.
        N (int): number of random tests to be performed. It impacts the running
        time directly.
        lines (int): number of lines of the dataframes of the test data.
        columns (int): number of columns of the dataframes of the test data.
        N_GRAPHICAL (int): Number of graphical tests to be performed.

    Returns:
        None: no exceptions should be raised if tests are correctly performed.

    Examples:
        >>> import data_utilities as du
        >>> du.test()

    """
    # TODO: implement the label variable.
    # TODO: implement the verbose variable.

    # Initial definitions.
    text_result = unittest.TextTestRunner(verbosity=100)

    # Resets values according to arguments:
    test_size_parameters = {
        'verbose': verbose,
        'N': N,
        'lines': lines,
        'columns': columns,
        'N_GRAPHICAL': N_GRAPHICAL,
    }
    for attr in test_size_parameters.keys():
        setattr(TestDataUtilitiesTestCase, attr, test_size_parameters[attr])

    # Filter test cases.
    test_objects = list()
    for module in (tpu, tmu, tpyu, ts):
        for defined_object in dir(module):
            defined_object = getattr(module, defined_object)  # str -> object
            try:
                if issubclass(defined_object, unittest.TestCase):
                    test_objects.append(defined_object)  # append classes
            except TypeError:
                pass

    # Initialize test instances.
    test_suite = unittest.TestSuite()
    test_suite.addTests(map(
        unittest.defaultTestLoader.loadTestsFromTestCase,
        test_objects))
    text_result.run(test_suite)

    return None
