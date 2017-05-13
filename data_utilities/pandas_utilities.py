"""Pandas utilities for common data management procedures.

All the functions should follow matplotlib, pandas and numpy's guidelines:

    Pandas:
        (1) Return a copy of the object; use keyword argument 'inplace' if
        changing is to be done inplace.

    Numpy:
        (1) Use the 'size' or 'shape' interface to determine the size of
        arrays.

    This module:
        (1) Functions should work out of the box whenever possible (for example
        for creating dataframes).

"""

import itertools
import pandas as pd
import numpy as np
import encodings
import io
import zipfile
import string
import re
import warnings

try:
    from unidecode import unidecode
except ImportError:
    warnings.warn("No module named unidecode", category=ImportWarning)

    def unidecode(x):
        """Mock unidecode function."""
        return x


def series_to_ascii(series):
    """
    Change columns to lowercase strings inplace.

    Arguments:
        series (pandas.Series): series to be modified.

    Returns:
        pandas.Series: series with lowercase and no symbols.

    """
    series = series.copy(True)
    series = series.apply(unidecode)
    series = series.str.lower()
    series = series.str.replace('[^a-zA-Z0-9_]', '_')

    return series


def discover_csv_encoding(
        csvpath,
        stop_on_first_found_encoding=True,
        validate_strings=(),):
    u"""Given a csv file path discover a list of encodings."""
    ALL_ENCODINGS = set(encodings.aliases.aliases.values())
    # ALL_ENCODINGS = set(encodings.aliases.aliases.keys())
    PRIORITY_ENCODINGS = set(['utf-8', 'cp1250'])

    list_of_working_encodings = []
    with open(csvpath, 'rb') as f:
        loaded_csv = f.read()
    for one_enc in PRIORITY_ENCODINGS | ALL_ENCODINGS:
        try:
            df = pd.read_csv(
                io.StringIO(loaded_csv.decode(one_enc)),
                encoding=one_enc,
                low_memory=False)
        # except UnicodeDecodeError:
            # pass
        # except LookupError:
            # pass
        # TODO: fix this general expression # noqa
        except Exception:  # noqa
            continue
        if validate_strings:
            for column, val_string in validate_strings:
                try:
                    # Sum will be greater than 0 if value is found.
                    # Thus it will evaluate to false.
                    # If no value is found then the loop is broken.
                    if not df.loc[:, column].str.contains(val_string).sum():
                        print(one_enc)
                        print(df.loc[:, column].tail(10))
                        print(df.loc[:, column].str.contains(val_string).sum())
                        print(10*'-')
                        break
                except KeyError:
                    # raise KeyError(
                        # 'the label [{0}] was not found. '
                        # 'The columns are: {1}'.format(
                            # column,
                            # sorted(df.columns.tolist())))
                    break
            # If list is exhausted then go inside else statement.
            else:
                list_of_working_encodings.append(one_enc)
        else:
            list_of_working_encodings.append(one_enc)
        if stop_on_first_found_encoding and list_of_working_encodings:
            break
    return list_of_working_encodings


def load_csv_from_zip(zippath, *args, **kwargs):
    u"""Load a csv file inside a zip file.

    Arguments:
        zippath (str): a valid file path leading to a zip file with a single
        csv file in it.

    Returns:
        DataFrame: a pandas dataframe loaded from the zipfile.

    """
    with zipfile.ZipFile(zippath, 'r') as zipfileobj:
        zip_file_list = zipfileobj.namelist()
        if len(zip_file_list) == 1:
            with zipfileobj.open(zip_file_list[0], 'r') as csvfile:
                return pd.read_csv(csvfile, **kwargs)
        else:
            raise ValueError('Zipfile conains more than one file.')


def object_columns_to_category(df, cutoff=0.1, inplace=False):
    u"""Transform object columns into categorical columns.

    Arguments:
        df (pandas.DataFrame): The pandas dataframe with object columns to be
        transformed into categorical.
        cutoff (float):

    Returns:
        pandas.DataFrame: The input df with object columns transformed into
        categorical.

    """
    # Inplace handling.
    if not inplace:
        df = df.copy(deep=True)
    # Extreme cases handling:
    if df.shape[0] == 0:
        if inplace:
            return None
        else:
            return df
    object_columns = (df.dtypes == object)
    if any(object_columns):
        object_columns = filter(
            lambda x: True
            if len(df[x].unique()) / df[x].shape[0] < cutoff
            else False,
            df.loc[:, object_columns])
        for column in object_columns:
            # Null values cause trouble later because of categories and
            # plotting with seaborn.
            df[column] = df[column].fillna('nan').astype('category')
    # Inplace handling.
    if inplace:
        return None
    else:
        return df


def rename_columns_to_lower(df):
    u"""Rename columns to lowercase and only the symbol '_'."""
    allowed_strings = string.ascii_lowercase + '_'
    df.columns = list(
        map(lambda x: re.sub(
            '[^' + allowed_strings + ']',
            '_',
            x.lower()),
            df.columns))


def read_string(string, **kwargs):
    u"""Read string and transform to a DataFrame."""
    try:
        kwargs['sep']
    except KeyError:
        kwargs['sep'] = '\s+'
    return pd.read_csv(
        io.StringIO(string),
        **kwargs)


def find_components_of_array(x, y, atol=1e-5, assume_int=False):
    """Find the components of x which compose y.

    Find the multiplier of each x column that results in y (some multipliers
    can be zero).

    Use case: suppose you receive a spreadsheet or dataframe whose columns are
    compositions of other columns. This function could then clarify which
    columns are compositions of other columns

    Arguments:
        x (pandas.DataFrame): Dataframe comprised of numeric columns that
        multiplied by some weight results in y.
        y (pandas.Series): Numeric series that is a linear combination of x's
        columns.

    Returns:
        dict: A dictionary where the keys are the columns that compose y and
        the values are the multipliers of x.

    Example:
        >>> x1 = 5 * np.arange(200)
        >>> x2 = 3 * np.arange(10, 210)
        >>> x3 = np.random.normal(size=200)
        >>> x = pd.DataFrame(data={'x1': x1, 'x2': x2, 'x3': x3})
        >>> y = pd.Series(x1 * 7 + x2 * 200)
        >>> find_components_of_array(x, y, assume_int=True) == {'x1': 7.0, 'x2': 200.0}  # noqa
        True
        >>>

    """
    # Borderline cases.
    # y is comprised of zeros.
    if np.all(y == 0):
        return dict()

    # Setup objects.
    dataframe = pd.concat((x, y), axis=1)
    original_dataframe = dataframe.copy(deep=True)

    # Sampling procedure should be improved. If columns are sparse frequently
    # they will raise a Singular Matrix error.
    is_not_zero = (dataframe != 0).all(axis=1)

    # TODO: need a new filter to drop duplicates. Some columns are just
    # identical to others on most values. Those cannot be picked up for
    # comparison otherwise they will just cancel each other.
    is_first_occurrence_column = (
        ~ dataframe.apply(pd.Series.duplicated)).all(axis=1)
    is_first_occurrence_row = (
        ~ dataframe.T.apply(pd.Series.duplicated).any())
    is_first_occurrence = (is_first_occurrence_row &
                           is_first_occurrence_column)

    # Remove duplicates from dataframes, do not allow zeros to cause
    # Singularity Errors and remove second occurrences to avoid equal
    # columns/variables.
    dataframe = dataframe.loc[(is_not_zero & is_first_occurrence), :].dropna()

    nlines, ncols = dataframe.shape
    nvariables = ncols - 1
    if ncols > nlines:
        raise ValueError(
            "System is under determined: {0:d} variables and {1:d} "
            "equations.".format(nvariables, nlines))

    sample = np.random.choice(dataframe.index, nvariables, replace=False)

    a = dataframe.ix[sample,
                     0:nvariables]
    b = dataframe.ix[sample, -1]
    # TODO: fix Singular Matrix problems (use stone data set). Some columns are
    # just equal to others; that happens when columns are sparse.
    # Approach: Transpose, drop duplicates then transpose back.

    # Compute the solution of the reduced linear system.
    solution = np.linalg.solve(a, b)

    # Now check that the results satisfied sumabs(y_calculated - y) == 0.
    c = np.multiply(original_dataframe.iloc[:, 0:nvariables],
                    solution).sum(axis=1)
    if np.isclose(
            np.abs(c - y.values).sum(),
            0.0,
            atol=atol):
        if assume_int:
            solution = np.rint(solution)
            solution_mask = (solution != 0)
            result = dict(zip(x.columns[solution_mask],
                              solution[solution_mask]))
        else:
            result = dict(zip(x.columns, solution))
        return result
    # TODO: calculate error
    # show error
    # return


def _construct_dataframe(shape, dict_of_functions):
    """Build a dataframe with a given ship from a dictionary of functions."""
    rows, columns = shape
    n_keys = len(dict_of_functions)
    data_dictionary = dict()
    for i, func_key in zip(range(columns), itertools.cycle(dict_of_functions)):
        data_dictionary[func_key + '_' +
                        str(i//n_keys)] = dict_of_functions[func_key]()

    return pd.DataFrame(data=data_dictionary)


def dummy_dataframe(
        shape=None,
        series_categorical=None,
        series_float=None,
        series_int=None,
        series_object=None):
    """Create an out-of-the-box dataframe with different datatype series."""
    # TODO: implement a boolean series.
    # Default value.
    if shape is None:
        rows, columns = (1000, 5)
    elif isinstance(shape, int):
        rows, columns = (shape, 5)
    else:
        rows, columns = shape

    # Requirements:
    #   a) unique 10 objects when n -> inf
    #   b) 100 objects when n = 10e4
    #   c) 3 objects when n = 3
    if series_categorical is None:
        n_objs = int(7e-4 * rows + 3) if rows <= 10000 else 10
        categories = list(
            itertools.islice(
                map(lambda x: ''.join(x),
                    itertools.product(string.ascii_lowercase,
                                      string.ascii_lowercase)),
                n_objs))

    def f_bool(): return np.random.randint(0, 1 + 1, size=rows, dtype=bool)

    def f_categorical(): return pd.Series(
        np.random.choice(categories, size=rows)).astype('category')

    def f_float(): return np.random.normal(size=rows)

    def f_int(): return np.arange(rows)

    def f_object(): return np.random.choice(categories, size=rows)

    dict_of_functions = {
        'bool': f_bool,
        'categorical': f_categorical,  # TODO: change to category.
        'float': f_float,
        'int': f_int,
        'object': f_object}

    return _construct_dataframe((rows, columns), dict_of_functions)


def statistical_distributions_dataframe(shape=None):
    """Create an out-of-the-box dataframe with common distrubutions.

    Arguments:
        n (int): number of rows in the dataframe.

    Returns:
        dataframe: (pandas.DataFrame): The dataframe of the common statistical
    distributions.

    Examples:
        >>> True
        True
        >>> # TODO:


    """
    if shape is None:
        rows, columns = (1000, 4)
    elif isinstance(shape, int):
        rows, columns = (shape, 4)
    else:
        rows, columns = shape

    def chisq(): return np.random.chisquare(5, size=rows)

    def stdnorm(): return np.random.standard_normal(size=rows)

    def logistic(): return np.random.logistic(size=rows)

    def rnd(): return np.random.random(size=rows)

    dict_of_functions = {
        'chisq': chisq,
        'stdnorm': stdnorm,
        'logistic': logistic,
        'rnd': rnd}

    return _construct_dataframe((rows, columns), dict_of_functions)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
