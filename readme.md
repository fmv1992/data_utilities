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

* Add test to every function.
    - Current coverage: 56%

* Move changelog and todo sections to separate files
  (https://github.com/pypa/sampleproject)

* ~~Setup TravisCI, add stickers of TravisCI and coverage of functions with
  tests.~~

# Changelog

#### Version 1.2.6

* Other
    * Stable version uploaded to pypi

#### Version 1.2.1 to 1.2.5

* Other
    * Development versions uploaded to pypi

#### Version 1.2.0

* Other
    * Stable version uploaded to pypi

#### Version 1.2.1 to 1.2.5

* Other
    * Development versions uploaded to pypi

#### Version 1.2.0

<!---
* `matplolib_utilities`
    * A

* `pandas_utilities`
    * A

* `python_utilities`
    * A
-->

* Other
    * Added package to PyPA as `data_utilities`
    * Added a test method to the package:
        python3 -c `import data_utilities as du; du.test()`

#### Version 1.1.0

* Improved `histogram_of_dataframe` function: added a textbox with summary
  statistics.

* Added a function to scale axes axis (`scale_axes_axis`).

* Added a colorbar argument to `plot_3d`.

* Label containers now return a list of matplotlib Text objects.

* Added a boolean column to `dummy_dataframe`.

* Added a test module for other libraries: `matplotlib_utilities` and
  `python_utilities`.

* Cleaned up the code.

#### Version 1.0.0

* Incompatible changes: the two utility functions to create dummy dataframes
  now use a keyword argument 'shape' instead of 'n' or 'rows' and 'columns' to
  resemble the numpy interface.

#### Version 0.1.1

* Added a convenience function (`statistical_distributions_dataframe`) of
  variable size initialized with some common statistical distributions.

* Added a test module which allows parametrized tests via the `TestMetaClass`.

* Added a test module for the `find_components_of_array` function.

#### Version 0.0.1

* First commit.

