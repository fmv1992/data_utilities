# Changelog

#### Version 1.2.8

* **Added `sklearn_utilities`**.
* Improved tests customization in `du.test`.
* Greatly improved documentation to `matplotlib_utilities`.
* Greatly expanded `pandas_utilities` functions.
* Improved tests as a whole.

#### Version 1.2.7

* Greatly improved `matplotlib_utilities` module.

* Removed dependency with `unidecode` module.

* tests: enabled parametrized tests invocations such as:  
  `python3 -c "import data_utilities as du; du.test(label='fast')"`  
  `python3 -c "import data_utilities as du; du.test(N=500)"`

* Add a test for the `test_support` file itself.

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
