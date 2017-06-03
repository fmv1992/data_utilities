https://help.github.com/articles/changing-a-commit-message/

## Tag wasteful commit messages with '__'

## Refactor code on matplotlib utilities

### What does matplotlib utilities need?

* `create_test_cases`: create the right number of test data cast into the right
  data type.
* [X] ``: an all figures attribute
    * [X] ``: an attribute with 2d figures.
        * ``: a function to generate random 2d data (can be the same as 3d)
        * histogram
        * lines
        * scatter
    * [X] ``: an attribute with 3d figures.
        * ``: a function to generate random 3d data (can be the same as 2d)
        * barplot
            ``: a function to generate random barplot functions
        * scatter
            ``: a function to generate random scatter functions
* ``: a random data generation function to feed into create_test_cases
    * 2d -> xy
    * 3d -> xyz
    * optional scaling of vectors
    * maybe create an ordered version
