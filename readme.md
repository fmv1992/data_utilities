[![Build Status](https://travis-ci.org/fmv1992/data_utilities.svg?branch=master)](https://travis-ci.org/fmv1992/data_utilities)

# Data Utilities

Data utilities library focused on machine learning and data analysis.

The library relies upon python's scientific/numeric stack to expand their
capabilities. The dependecies are:
* numpy
* scipy
* pandas
* matplotlib
* seaborn
* scikit-learn

Optional depencies are:
* XGBoost
* deap

Highlights are:
* matplotlib_utilities: out-of-the-shelf data description with
  `histogram_of_dataframe`.
* pandas_utilities: easier dataframe preparation with
  `rename_columns_to_lower`, `categorical_serie_to_binary_dataframe`,
  `balance_ndframe` and `get_numeric_columns`.
* sklearn_utilities: multiprocessing and persistance support for hyper
  parameter grid search, both exhaustive and using a genetic algorithmic
  approach; convenience functions to the XGBoost module.

And much more.

# Organization and files

    ./data_utilities
    ├── __init__.py
    ├── matplotlib_utilities.py
    ├── pandas_utilities.py
    ├── python_utilities.py
    ├── sklearn_utilities
    │   ├── evolutionary_grid_search.py
    │   ├── grid_search.py
    │   └── __init__.py
    └── tests
        ├── __init__.py
        ├── test_matplotlib_utilities.py
        ├── test_pandas_utilities.py
        ├── test_python_utilities.py
        ├── test_sklearn_utilities.py
        └── test_support.py

Each of python's significant data modules has its own set of functions.
Optional dependencies functions are interspersed throughout the code.

This module does not intend to create its own API or standards. Instead each of
the utilities module should follow the guidelines and APIs provided by the
parent module.

Note: This is a primitive project. Expect backwards incompatible changes as I
figure out the best way to to develop the utilities.

# What's new

* xxx TODO xxx

# Development guidelines

* Coding style: [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliant.
* Docstrings: [google docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
* Before commiting new versions do a test for different versions of python3:
    * python3.4
    * python3.5
    * python3.6
    * (newer versions)
    * Rationale: even though stability is expected between python versions some
      changes occur. See for instance that on commit v1.2.8 (60573d7) there was
      as unexpected import error on python34 but not on python36.
