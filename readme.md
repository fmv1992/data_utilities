# Data Utilities

This module provides some helper functions and conveniences for working with
data analysis in python.

It depends on:

- Numpy
- Scipy
- Pandas
- Matplotlib
- Seaborn

# Organization and files

    .
    ├── data_utilities
    │   ├── matplotlib_utilities.py
    │   ├── pandas_utilities.py
    │   └── python_utilities.py
    ├── readme.md
    └── tests
        └── test.py

Each of python's significant data modules has its own set of helper functions.

This module does not intend to create its own API or standards. Instead each of
the utilities module should follow the guidelines and APIs provided by the
parent module.

Note: This is a primitive project. Expect backwards incompatible changes as I
figure out the best way to to develop the utilities. Use at your own risk :)

# TODO

- Setup TravisCI, add stickers of TravisCI and coverage of functions with
  tests.

- Uniformize interface for generating dummy dataframes. The interface of numpy
  should be a starter (`size=` or `shape=` arguments).

- Add test to every function.

# Changelog

#### Version 0.1.1

- Added a convenience function (`statistical_distributions_dataframe`) of
  variable size initialized with some common statistical distributions.

- Added a test module which allows parametrized tests via the `TestMetaClass`.

- Added a test module for the `find_components_of_array` function.

#### Version 0.0.1

- First commit.
