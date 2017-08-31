# My structured development to matplotlib utilities

## Tag wasteful commit messages with '__'

## Refactor code on matplotlib utilities

* Understand that there are functions that **generate** figures and functions
  that **modify** figures.
    * Two types of functions:
        * Functions that generate figures: generate figures then append them to
        figures and figures_xd
        * Functions that modify figures: should then run on an existing set of
        'core' figures
    * 

### Test lifecycle: Functions that generate figures

* ***Attribute: `created_figures`***:
    * Initially: an empty list
    * Grow: Append figure objects on each test
    * End: save figures to files

### Test lifecycle: Functions that modify figures

* Call proper function to generate `figure_xd`
* ***Attribute: `created_figures`***:
    * Initially: an empty list
    * Grow: Append figure objects on each test
    * End: save figures to files

### What does matplotlib utilities need?

* `create_test_cases`: create the right number of test data cast into the right
  data type.
* [X] `XXX`: an all figures attribute
    * [X] `XXX`: an attribute with 2d figures.
        * `XXX`: a function to generate random 2d data (can be the same as 3d)
        * histogram
        * lines
        * scatter
    * [X] `XXX`: an attribute with 3d figures.
        * `XXX`: a function to generate random 3d data (can be the same as 2d)
        * barplot
            `XXX`: a function to generate random barplot functions
        * scatter
            `XXX`: a function to generate random scatter functions
* `XXX`: a random data generation function to feed into create_test_cases
    * 2d -> xy
    * 3d -> xyz
    * optional scaling of vectors
    * maybe create an ordered version
