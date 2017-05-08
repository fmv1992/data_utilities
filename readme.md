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
    │   └── tests
    │       ├── __init__.py
    │       ├── test_matplotlib_utilities.py
    │       ├── test_pandas_utilities.py
    │       ├── test_python_utilities.py
    │       └── test_support.py
    ├── license
    ├── MANIFEST.in
    ├── readme.md
    └── setup.py

Each of python's significant data modules has its own set of helper functions.

This module does not intend to create its own API or standards. Instead each of
the utilities module should follow the guidelines and APIs provided by the
parent module.

Note: This is a primitive project. Expect backwards incompatible changes as I
figure out the best way to to develop the utilities. Use at your own risk :)

# TODO

## Tests

* Add test to every line of code.
    - Current coverage: 69%

* Output images if `save_figures == True` to a tempfolder

## Other

* Move changelog and todo sections to separate files
  (https://github.com/pypa/sampleproject)

* Create a helper function to allow for easy plotting of 20-30 data points to
  be easily distinguishable on the map. Cycling thru colors, markers and
  dash/lines is a good way to start.
    * http://seaborn.pydata.org/tutorial/color_palettes.html is a good place to
      start.

* Solve all open "XXX"s and "TODO"s

* Add a way to create release versions number unequivocally (instead of doing
  the analysis of backwards compatibility myself)


# Changelog

#### Version 1.2.7

* Greatly improved `matplotlib_utilities` module

* Removed dependency with `unidecode` module

* tests: enabled parametrized tests invocations such as:  
  `python3 -c "import data_utilities as du; du.test(label='fast')"`  
  `python3 -c "import data_utilities as du; du.test(N=500)"`

* Add a test for the `test_support` file itself

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

