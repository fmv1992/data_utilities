# Data Utilities

This module intends to provide some helper functions and conveniences for working with data analysis in python.

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

This module does not intend to create its own API or standards. Instead each of the utilities module should follow the guidelines and APIs provided by the parent module.

Note: This is a primitive project. Expect backwards incompatible changes as I figure out the best way to to develop the utilities. Use at your own risk :)

# TODO

- Add test to every functions.

# Changelog

## Version 0.0.1

- First commit.
