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

# Changelog

#### Version 1.1.0

- Improved `histogram_of_dataframe` function: added a textbox with summary
  statistics.

- Added a boolean column to `dummy_dataframe`.

- Added a test module for other libraries matplotlib_utilities.

- Cleaned up the code.

#### Version 1.0.0

- Incompatible changes: the two utility functions to create dummy dataframes
  now use a keyword argument 'shape' instead of 'n' or 'rows' and 'columns' to
  resemble the numpy interface.

#### Version 0.1.1

- Added a convenience function (`statistical_distributions_dataframe`) of
  variable size initialized with some common statistical distributions.

- Added a test module which allows parametrized tests via the `TestMetaClass`.

- Added a test module for the `find_components_of_array` function.

#### Version 0.0.1

- First commit.

# TODO

- Add test to every function.
    - Current coverage: 42%

- Improved test modules structures. Now all test cases are run from the same
  data (possibly costly operation).

- Aggregated `dummy_dataframe` and `statistical_distributions_dataframe` in the
  same constructor.

- Aggregated `dummy_dataframe` and `statistical_distributions_dataframe` tests
  in the same constructor

- Setup TravisCI, add stickers of TravisCI and coverage of functions with
  tests.

- ~~Add a boolean column to `dummy_dataframe`.~~

- ~~Uniformize interface for generating dummy dataframes. The interface of numpy
  should be a starter (`size=` or `shape=` arguments).~~
