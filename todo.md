# TODO

## Evolutionary Grid Search

* Hash of dataset should be basename instead of full path (copy paste folder then run new model causes grid to be re evaluated)

* Add verbosity to grid search

* ***Clone each estimator before using; do not interfere with the original
  object***.
* Do not save x and y for every individual of the population.
* Make it parallel.
* Handle the minimize/maximize switching on optimization.
* Improve resemblance to GridSearchCV attributes.

## Tests

* Put all XGBoost related tests under `TestXGBoostFunctions`.
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

* Add a `random_forest_get_learning_curve` function. This function should:
    1. Start with a huge forest with huge depth.
    1. Sort the decision trees according to the evaluation metric.
    1. Plot different line curves for each pruned decision tree with a given
       depth.  
       OR  
       Plot the difference between train and test results for the metric.  
       That means that on the x axis you have the number of trees. On the
       y axis you have the evaluation metric.
* Create a helper function to allow for easy plotting of 20-30 data points to
  be easily distinguishable on the map. Cycling thru colors, markers and
  dash/lines is a good way to start.
    * http://seaborn.pydata.org/tutorial/color_palettes.html is a good place to
      start.
* Add a way to create release versions number unequivocally (instead of doing
  the analysis of backwards compatibility myself).
* Solve all open "XXX"s and "TODO"s
