"""Data analysis helper module written in python."""
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
import os

__version__ = '1.2.6'


def test(*a, **kw):
    """Module level test function.

    Run all tests using the unittest module.

    Created based on the same architecture as the scipy test function defined on
    their __init__.py.

    """
    base_dir = os.path.dirname(__file__)
    # TODO: adjust verbosity parameter.
    text_result = unittest.TextTestRunner(verbosity=100)
    test_suite = unittest.TestLoader().discover(
        os.path.join(base_dir, 'tests'))
    text_result.run(test_suite)
