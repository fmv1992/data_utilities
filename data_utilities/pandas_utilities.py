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

import encodings
import io
import itertools
import numpy as np
import pandas as pd
import re
import string
import warnings
import zipfile

try:
    from unidecode import unidecode
except ImportError:
    warnings.warn("No module named unidecode", category=ImportWarning)

    def unidecode(x):
        """Mock unidecode function."""
        return x


# Section on dataframe loading and creating.
# ------------------------------------------
def _construct_dataframe(shape, dict_of_functions):
    """Build a dataframe with a given ship from a dictionary of functions."""
    rows, columns = shape
    n_keys = len(dict_of_functions)
    data_dictionary = dict()
    for i, func_key in zip(range(columns), itertools.cycle(dict_of_functions)):
        data_dictionary[func_key + '_' +
                        str(i // n_keys)] = dict_of_functions[func_key]()

    return pd.DataFrame(data=data_dictionary)


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


def dummy_dataframe(
        shape=None,
        series_categorical=None,
        series_float=None,
        series_int=None,
        series_object=None):
    """Create an out-of-the-box dataframe with different datatype series."""
    # TODO: rework function. It makes no sense to specify series and all into a
    # function when one could just append those and create a dataframe.

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


def read_string(string, **kwargs):
    u"""Read string and transform to a DataFrame."""
    try:
        kwargs['sep']
    except KeyError:
        kwargs['sep'] = '\s+'
    return pd.read_csv(
        io.StringIO(string),
        **kwargs)


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
                return pd.read_csv(csvfile, *args, **kwargs)
        else:
            raise ValueError('Zipfile conains more than one file.')


# Section on dataframe/series mutation (with inplace option).
# -----------------------------------------------------------
def balance_ndframe(ndframe_obj,
                    column_to_balance=None,
                    ratio=1,
                    inplace=False):
    """Balance a given ndframe_obj.

    Balance a given ndframe_obj/series considering the column_to_balance
    column. This column must be binary.

    Arguments:
        ndframe_obj (pandas.ndframe_obj): ndframe_obj/series to be balanced.
        column_to_balance (str): column name to be balanced.
        ratio (float): ratio of the most frequent value to the least frequent
        value.

    Returns:
        pandas.ndframe_obj: the balanced ndframe_obj

    Example:
        >>> np.random.seed(0)
        >>> series = pd.Series(np.random.choice(list('abc'), size=1000))
        >>> series[series == 'c'] = 'a'
        >>> balanced = balance_ndframe(series)
        >>> vc = balanced.value_counts()
        >>> np.isclose(vc[0] / vc[1], 1)
        True

    """
    # Set the same object for dataframe input or series input.
    if column_to_balance is not None:
        series_to_balance = ndframe_obj[column_to_balance]
    else:
        series_to_balance = ndframe_obj

    # Store original ndframe_obj index.
    index = series_to_balance.index

    # Count values to enable filtering.
    value_counts = series_to_balance.value_counts(ascending=False)
    value_counts = value_counts[value_counts > 0]

    # Unpack most frequent and unfrequent values.
    vmax, vmin = value_counts.index.tolist()

    # Create filter based on frequency.
    samples_min_index = index[series_to_balance == vmin]
    samples_max_index = index[series_to_balance == vmax]
    samples_max_balanced_index = index[series_to_balance == vmax][
        0:int(ratio * len(samples_min_index))]

    # Assert that the given ratio is possible.
    max_ratio = len(samples_max_index) / len(samples_min_index)
    calculated_ratio = len(samples_max_balanced_index) / len(samples_min_index)
    assert ratio <= max_ratio, \
        'Given ratio is infeasible (given={0}, maximum={1}.'.format(
            ratio, max_ratio)
    assert np.isclose(ratio, calculated_ratio, rtol=1e-2), \
        'Calculated ratios differ: (given={0}, calculated={1}.'.format(
            ratio, calculated_ratio)

    # Create a view of the filtered series_to_balance.
    new_index = pd.Index(np.concatenate((samples_min_index,
                                         samples_max_balanced_index)))
    if inplace:
        complementar_index = index.drop(new_index)
        ndframe_obj.index.drop(complementar_index, inplace=True)
        return None
    else:
        return ndframe_obj.loc[new_index].copy()


# TODO: move this to other section.
def categorical_serie_to_binary_dataframe(series,
                                          keep_original_name=True,
                                          nan_code='nan',
                                          include_nan=False):
    """Transform a categorical serie into a binary dataframe.

    Arguments:
        series (pandas.Series): series TODO.

    Returns:
        pandas.DataFrame: dataframe TODO.

    """
    df = pd.DataFrame()
    for i, uvalue in enumerate(series.unique()):
        if keep_original_name:
            colname = str(uvalue)
        else:
            colname = str(i)
        bool_series = pd.Series(series == uvalue)
        bool_series.name = series.name + '_' + colname
        df = pd.concat((df, bool_series), axis=1)
    return df


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


def rename_columns_to_lower(
        df,
        allowed_strings=string.ascii_lowercase + '_' + string.digits):
    u"""Rename columns to lowercase and only the symbol '_'."""
    if not allowed_strings:
        allowed_strings = string.ascii_lowercase + '_' + string.digits
    df.columns = list(
        map(lambda x: re.sub(
            '[^' + allowed_strings + ']',
            '_',
            x.lower()),
            df.columns))


# Section on csv processing.
# --------------------------
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
                        print(10 * '-')
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


# Section on dataframe derivation
# -------------------------------
# (no inplace option, output is clearly from a different from than the initial
# dataframe).
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


def get_numeric_columns(dataframe):
    """Get numeric columns from a dataframe."""
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32',
                      'float64', 'bool']
    numeric_columns = (
        dataframe.dtypes.apply(str).isin(numeric_dtypes) == True)  # noqa
    numeric_columns = numeric_columns.index[numeric_columns]
    return numeric_columns


def get_non_numeric_columns(dataframe):
    """Get numeric columns from a dataframe."""
    numeric_columns = get_numeric_columns(dataframe)
    non_numeric_columns = sorted(set(dataframe.columns.tolist())
                                 - set(numeric_columns))
    return non_numeric_columns


def group_sorted_series_into_n_groups(series, n_groups=100):
    """Group sorted series into N groups.

    Useful to compute aggregation statistic in the divided groups.

    Input should be a sorted array.

    Examples:
        >>> from data_utilities import pandas_utilities as pu
        >>> from scipy.special import expit
        >>> np.random.seed(0)
        >>> x = pd.Series(np.random.normal(size=10000))
        >>> y_predicted_proba = expit(x.sort_values())
        >>> gb = pu.group_sorted_series_into_n_groups(y_predicted_proba)
        >>> gb.mean().iloc[:, 0].is_monotonic
        True

    """
    # TODO: check the necessity of this.
    # Check that the array is sorted.
    # assert series.is_monotonic, "Provided series is not monotonic."

    alen = len(series)
    repeat_times = alen // n_groups
    remainder = alen % n_groups

    repeat_array = np.repeat(np.arange(n_groups), repeat_times)
    if remainder == 0:
        combined_array = repeat_array
    else:
        remainder_array = np.arange(remainder)
        combined_array = np.concatenate(
            (np.stack(
                (repeat_array[:remainder], remainder_array), axis=1).flatten(),
             repeat_array[remainder:]))

    return pd.DataFrame(
        {series.name: series, 'groups': combined_array}).groupby('groups')


def categorize_n_most_frequent(series, n, other_name='other', inplace=False):
    """Categorize the most frequent values in a series.

    Set the n most frequent occurences in a dataframe and set all other values
    to 'other.'

    Arguments:
        series (pandas.Series): series to be modified.
        n (int): the n most frequent groups to preserve.
        other_name (str): The name to set to other categories/less frequent
        categories.

    Examples:
        >>> import data_utilities as du
        >>> du.set_random_seed(0)
        >>> a = np.random.choice(tuple('abcd'),
        ...     size=10000,
        ...     p=(0.5, 0.3, 0.1, 0.1))
        >>> # Category a will amount to ~ 50% of occurences.
        >>> s1 = pd.Series(a)
        >>> s2 = categorize_n_most_frequent(s1, 1)
        >>> vc = s2.value_counts()
        >>> np.isclose(vc[0] / vc[1], 1.02, atol=1e-2)
        True


    """
    if not inplace:
        series = series.copy(True)
    else:
        raise NotImplementedError
    # TODO: cover the case of serie which is already categorical.
    vc = series.value_counts(ascending=True)
    categories = set(vc.iloc[-n:].index)
    series.loc[~ series.isin(categories)] = other_name
    if not inplace:
        return series.astype('category')


# Section: deprecated functions.
# ------------------------------
def series_to_ascii(series):
    """Change columns to lowercase strings inplace.

    Arguments:
        series (pandas.Series): series to be modified.

    Returns:
        pandas.Series: series with lowercase and no symbols.

    """
    warnings.warn("Function will be deprecated because it is not used.",
                  category=DeprecationWarning)
    series = series.copy(True)
    series = series.apply(unidecode)
    series = series.str.lower()
    series = series.str.replace('[^a-zA-Z0-9_]', '_')

    return series


if __name__ == '__main__':
    import doctest
    doctest.testmod()
