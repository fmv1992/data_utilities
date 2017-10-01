[![Build Status](https://travis-ci.org/fmv1992/data_utilities.svg?branch=master)](https://travis-ci.org/fmv1992/data_utilities)

# Data Utilities

This module provides some helper functions and conveniences for working with
data analysis in python.

It depends on:

* Numpy
* Scipy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

# Organization and files

    .
    ├── data_utilities
    │   ├── __init__.py
    │   ├── matplotlib_utilities.py
    │   ├── pandas_utilities.py
    │   ├── python_utilities.py
    │   ├── sklearn_utilities
    │   │   ├── grid_search.py
    │   │   └── __init__.py
    │   └── tests
    │       ├── __init__.py
    │       ├── test_matplotlib_utilities.py
    │       ├── test_pandas_utilities.py
    │       ├── test_python_utilities.py
    │       ├── test_sklearn_utilities.py
    │       └── test_support.py
    ├── LICENSE
    ├── MANIFEST.in
    ├── readme.md
    └── setup.py

Each of python's significant data modules has its own set of helper functions.

This module does not intend to create its own API or standards. Instead each of
the utilities module should follow the guidelines and APIs provided by the
parent module.

Note: This is a primitive project. Expect backwards incompatible changes as I
figure out the best way to to develop the utilities.

# What's new

* **Added `sklearn_utilities`**.
* Improved tests customization in `du.test`.
* Greatly improved documentation to `matplotlib_utilities`.
* Greatly expanded `pandas_utilities` functions.
* Improved tests as a whole.

# Development guidelines

* Coding style: [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliant.
* Docstrings: [google docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

* Support first the test interface of numpy:

        `python3 -c "import data_utilities as du; du.test()"`
  and then the unittest interface:

        `python3 -m unittest discover -vvv data_utilities/tests`
