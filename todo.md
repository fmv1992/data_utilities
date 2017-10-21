# TODO

## Tests

* Add travis/ci support for Windows (multiprocessing is not playing nice on
  Win).
* Add test to every line of code.
    - Current coverage: 67%
* Output images if `save_figures == True` to a tempfolder (instead of `/tmp`).
    * Including doctests.
* Support first the test interface of numpy:
        `python3 -c "import data_utilities as du; du.test()"`
  and then the unittest interface:
        `python3 -m unittest discover -vvv data_utilities/tests`

## Other

* Create a helper function to allow for easy plotting of 20-30 data points to
  be easily distinguishable on the map. Cycling thru colors, markers and
  dash/lines is a good way to start.
    * http://seaborn.pydata.org/tutorial/color_palettes.html is a good place to
      start.
* Add a way to create release versions number unequivocally (instead of doing
  the analysis of backwards compatibility myself).
* Solve all open "XXX"s and "TODO"s
