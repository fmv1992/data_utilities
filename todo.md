# TODO

## Tests

* Add travis/ci support for windows (multiprocessing is not playing nice on
  win).
* Add test to every line of code.
    - Current coverage: 67%
* Output images if `save_figures == True` to a tempfolder (instead of `/tmp`).
* Improve testing on PersitentGrid (a bug was found by a different project).

## Other

* ~~Move changelog and todo sections to separate files.~~
  (https://github.com/pypa/sampleproject)
* Create a helper function to allow for easy plotting of 20-30 data points to
  be easily distinguishable on the map. Cycling thru colors, markers and
  dash/lines is a good way to start.
    * http://seaborn.pydata.org/tutorial/color_palettes.html is a good place to
      start.
* Add a way to create release versions number unequivocally (instead of doing
  the analysis of backwards compatibility myself).
* Solve all open "XXX"s and "TODO"s
