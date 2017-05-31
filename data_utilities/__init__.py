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


def test(label='fast',
         verbose=False,                 # Must match the default values for
         n_tests=5,                     # fast label.
         n_lines=50,
         n_columns=5,
         n_graphical_tests=3,
         save_figures=False,
         **kwargs_test_runner):
    """Module level test function.

    Run tests using the unittest module. Both 'numpy style' and unittest
    invocations should work:

    `python3 -m unittest discover -vvv data_utilities/tests`
        (not fully supported as of yet)
    `python3 -c "import data_utilities as du; du.test()"`
        (fully supported as of yet)

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
        n_graphical_tests (int): Number of graphical tests to be performed.
        save_figures (bool): True to save images in test folder.

    Returns:
        None: no exceptions should be raised if tests are correctly performed.

    Examples:
        >>> import data_utilities as du
        >>> du.test(verbose=False)

    """
    # TODO: implement the 'fast' label and the default values of variables as
    # the default argments. So overriding any of them should only change the
    # changed variable in a simple way.

    # Default function parameters.
    default_function_parameters = {
        'verbose': False,
        'n_tests': 5,
        'n_lines': 50,
        'n_columns': 5,
        'n_graphical_tests': 3,
        'save_figures': False,
    }

    # Parse function parameters.
    function_parameters = {
        'verbose': verbose,
        'n_tests': n_tests,
        'n_lines': n_lines,
        'n_columns': n_columns,
        'n_graphical_tests': n_graphical_tests,
        'save_figures': save_figures,
    }
    # Update those parameters related to new keyword parameters.
    function_parameters = {
        k: v for k, v in function_parameters.items() if
        v != default_function_parameters[k]}

    # Enforce label variable.
    if label == 'fast':
        updated_function_parameters = default_function_parameters.copy()
        updated_function_parameters.update(function_parameters)
    elif label == 'full':
        updated_function_parameters = {
            'verbose': False,
            'n_tests': 100,
            'n_lines': 500,
            'n_columns': 50,
            'n_graphical_tests': 20,
            'save_figures': True,
        }
        updated_function_parameters.update(function_parameters)
    else:
        raise NotImplementedError(
            "label == '{}' is not implemented yet.".format(label))

    # Overwrite parsed variables.
    for attr in updated_function_parameters.keys():
        setattr(TestDataUtilitiesTestCase,
                attr,
                updated_function_parameters[attr])
    # Update data.
    TestDataUtilitiesTestCase.update_data()

    # Initial definitions.
    text_result = unittest.TextTestRunner(verbosity=verbose,
                                          failfast=True,  # TODO: remove this.
                                          **kwargs_test_runner)

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
