"""Matplotlib utilities for common plotting procedures.

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

from pprint import pprint
import itertools
import os
import random

from mpl_toolkits.mplot3d import Axes3D
from pandas_utilities import object_columns_to_category
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# pylama: ignore=W0611,D301


def plot_3d(serie):
    """Create a 3d-barchart axes for a given 2-level-multi-index series.

    Return a 3d axes object given a series with a multiindex with 2
    categorical levels.

    Arguments:
        serie (Series): the 2-level-index series to generate the plot.

    Returns:
        matplotlib.axes.Axes: the 3d axis object.

    Examples:
        >>> import itertools
        >>> import pandas as pd
        >>> fig = plt.figure()
        >>> s_index = pd.MultiIndex.from_tuples(                              \
            tuple(itertools.product(range(6), list('abc'))),                  \
        names=('x1', 'x2'))
        >>> s = pd.Series(data=np.arange(18), index=s_index)
        >>> ax = plot_3d(s)
        >>> fig.tight_layout()
        >>> fig.savefig('/tmp/{0}.png'.format('plot3d'), dpi=500)

    """
    # Create a copy of the list to avoid changing the original.
    serie = serie.copy(deep=True)
    serie.sort_values(inplace=True, ascending=False)
    # Set constants.
    # TODO: Make graphs on the same scale effective.
    # View:
    # http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to  # noqa
    SCALE_AXIS_DIST_FACTOR = 1  # This seemingly changes de width of the bar.
    SCALE_AXIS_DIST_TRESH = 1
    # Some groupby objects will produce a dataframe. Not nice going over duck
    # typing but oh well...
    # If it is a dataframe with one column then transform it to series.
    if isinstance(serie, pd.DataFrame) and serie.shape[1] == 1:
        serie = serie.ix[:, 0]

    # Error handling phase.
    # Track if index has correct shape.
    if len(serie.index.levshape) != 2:
        raise ValueError('The index level shape should '
                         'be 2 and it is {}.'.format(serie.index.levshape))
    # Check for duplicate indexes.
    if serie.index.duplicated().sum():
        serie = serie.groupby(level=serie.index.names).sum()
        if serie.index.duplicated().sum():
            raise ValueError('Series has duplicate values.')

    # Handling the index of the received serie.
    level1_index, level2_index = tuple(zip(*serie.index.get_values()))
    level1_index = sorted(set(level1_index))
    level2_index = sorted(set(level2_index))

    all_index_combinations = tuple(itertools.product(
        level1_index,
        level2_index))
    index_names = serie.index.names
    new_index = pd.MultiIndex.from_tuples(all_index_combinations,
                                          names=index_names)

    # This new dataframe has all combinations of possible indexes.
    new_dataframe = pd.Series(0, index=new_index, name=serie.name)
    serie = (serie + new_dataframe).fillna(0)

    # Generate the z values
    z_values = []
    for _, group in serie.groupby(level=1):
        z_values.append(group)
    z_values = np.hstack(z_values).ravel()

    # Starts manipulating the axes
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')

    # Create the axis and their labels
    xlabels = serie.index.get_level_values(index_names[0]).unique().tolist()
    ylabels = serie.index.get_level_values(index_names[1]).unique().tolist()
    xlabels = [''.join(list(filter(str.isalnum, str(value))))
               for value in xlabels]
    ylabels = [''.join(list(filter(str.isalnum, str(value))))
               for value in ylabels]
    x = np.arange(len(xlabels))
    if len(x) > SCALE_AXIS_DIST_TRESH:
        x = x * SCALE_AXIS_DIST_FACTOR
    y = np.arange(len(ylabels))
    if len(y) > SCALE_AXIS_DIST_TRESH:
        # TODO: fix this.
        y = y * SCALE_AXIS_DIST_FACTOR
    xlabels = [z.title() for z in xlabels]
    ylabels = [z.title() for z in ylabels]

    x_mesh, y_mesh = np.meshgrid(x, y, copy=False)

    ax.w_xaxis.set_ticks(x + 0.5/2.)
    ax.w_yaxis.set_ticks(y + 0.5/2.)

    # TODO: how does one create a scaled figure?
    # ax.set_aspect('equal', 'datalim')
    # ax.axis('scaled')
    ax.set_aspect('equal')

    ax.w_xaxis.set_ticklabels(xlabels)
    ax.w_yaxis.set_ticklabels(ylabels)

    # Color.
    values = np.linspace(0.2, 1., x_mesh.ravel().shape[0])
    colors = plt.cm.Spectral(values)

    # Create the 3d plot.
    ax.bar3d(x_mesh.ravel(), y_mesh.ravel(), z_values*0,
             dx=0.5, dy=0.5, dz=z_values,
             color=colors)

    # print(x)
    # print(y)
    # ax.autoscale_view()
    # ax.view_init(-64, 47)
    # TODO: the newxt two folling lines are key to success.
    xy = np.concatenate((x, y))
    ax.set_xlim3d(xy.min(), xy.max())
    ax.set_xlim3d(xy.min(), xy.max())
    # ax.set_xlim3d(x.min(), x.max())
    # ax.set_ylim3d(y.min(), y.max())
    # print(ax.get_xlim3d())
    # print(ax.get_ylim3d())
    # ax.set_zlim3d(0, 16)

    return ax


def label_container(axes,
                    containers=None,
                    string_formatting=None,
                    label_height_increment=0.01):
    u"""Attach text labels to axes.

    Arguments:
        axes (matplotlib.axes.Axes): Axes in which text labels will be added.
        containers (list): List of matplotlib.container.Container objects.
        string_fomratting (str): string that will be passed to str.format
        function.

    Returns:
        None

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib_utilities import label_container
        >>> fig = plt.figure(num=0, figsize=(12, 9))
        >>> ax = fig.add_subplot(1,1,1)
        >>> x = range(10)
        >>> y = range(10)
        >>> ax.bar(x, y)
        <Container object of 10 artists>
        >>> label_container(ax)
        >>> fig.tight_layout()
        >>> fig.savefig('/tmp/{0}.png'.format('label_container'))

    """
    # TODO: improve string formatting based on rect height.
    if containers is None:
        containers = axes.containers[0]
    height = np.fromiter(
        (x.get_height() for x in containers), float)
    if string_formatting is None:
        # Set standards strings for integers and floats.
        # TODO: Add a standard formatting for percentages.
        if np.all(np.equal(np.mod(height, 1), 0)):
            string_formatting = '{0:d}'
            height = height.astype('int')
        else:
            string_formatting = '{0:1.1f}'

    height_increment = height.max() * label_height_increment
    for i, rect in enumerate(containers):
        label_height = height[i] + height_increment
        axes.text(rect.get_x() + rect.get_width()/2.,
                  label_height,
                  string_formatting.format(height[i]),
                  ha='center', va='bottom')
    return None


def histogram_of_categorical(x,
                             label_containers=False,
                             normalized_to_one=False,
                             *args,
                             **sns_distplot_kwargs):
    """Plot a histogram of categorical with sane defauts."""
    pass


def histogram_of_floats(x,
                        label_containers=False,
                        normalized_to_one=False,
                        *args,
                        **sns_distplot_kwargs):
    """Plot a histogram of float with sane defauts."""
    axes = sns.distplot(
        x,
        *args,
        **sns_distplot_kwargs)
    if label_containers:
        label_container(axes,
                        string_formatting='{0:1.1%}')
    return axes


def histogram_of_integers(x,
                          label_containers=False,
                          normalized_to_one=False,
                          *args,
                          **sns_distplot_kwargs):
    """Plot a histogram of integers with sane defauts.

    Arguments:
    Returns:
    Examples:

    """
    # TODO: this function is still in experimental mode.
    # print("Function not stable yet!".upper())

    THRESHOLD_TO_CONSIDER_FLOAT = 100
    unique = np.unique(x).shape[0]
    if unique > THRESHOLD_TO_CONSIDER_FLOAT:
        return histogram_of_floats(
            x,
            normalized_to_one=normalized_to_one,
            *args,
            **sns_distplot_kwargs)
    # TODO: Cover the case of less than treshold number of integers but with a
    # lot of spacing between them such as (0, 1, 2, 3, 5500, 15000).
    if x.max() - x.min() > THRESHOLD_TO_CONSIDER_FLOAT:
        unique_values = np.sort(x.unique())
        mask_values = dict(zip(unique_values, range(len(unique_values))))
        x = x.map(mask_values)
    if normalized_to_one:
        weights = np.ones_like(x)/len(x)
    else:
        weights = None
    xlabels = np.arange(x.min() - 2,
                        x.max() + 3)
    # Put default
    if 'hist_kws' not in sns_distplot_kwargs:
        sns_distplot_kwargs['hist_kws'] = dict()
        hist_kws = sns_distplot_kwargs['hist_kws']
    DEFAULT_HIST_KWARGS = {
                    'weights': weights,
                    'align': 'mid',
                    'rwidth': 0.5}
    # Update kwargs to matplotlib histogram which were not specified.
    for absent_key in filter(lambda x: x not in
                             hist_kws.keys(),
                             DEFAULT_HIST_KWARGS.keys()):
            hist_kws[absent_key] = DEFAULT_HIST_KWARGS[absent_key]
    axes = sns.distplot(
        x,
        bins=xlabels - 0.5,
        *args,
        **sns_distplot_kwargs)
    axes.set_xticks(xlabels)
    # If it is the case of having mapped the values.
    try:
        mask_values
        x = x.map({v: k for k, v in mask_values.items()})
        xlabels = np.concatenate((
            np.arange(x.min() - 2, x.min() - 1),
            np.sort(np.unique(x)),
            np.arange(x.max() + 1, x.max() + 3)))
    except NameError:
        pass
    if max(x) >= 100:
        rotation = -45
    else:
        rotation = 0
    axes.set_xticklabels(xlabels, rotation=rotation)

    return axes


def histogram_of_dataframe(dataframe,
                           output_path,
                           normalize=True,
                           weights=None,
                           *args,
                           **sns_distplot_kwargs):
    """Draw a histogram for each column of the dataframe.

    Arguments:
    Returns:
    Examples:
        >>> import pandas_utilities
        >>> dummy_df = pandas_utilities.dummy_dataframe(shape=200)
        >>> histogram_of_dataframe(dummy_df, '/tmp/')
    """
    # TODO: modular: do one thing well. Return a tuple of axes. Write a new
    # function to save a collection of axes to different figures. Remove
    # output_path.

    # TODO: stick to an interface. It would be good to use the same interface
    # as seaborn.

    # Implement weight argument.
    # TODO: Include minimum and maximum.
    #   minimum: sometimes max is too big and will obliterate minimum from df.
    #   maximum: likewise.

    # What to do with nans?
    # TODO: Create a statement about nans.

    # Should we cast Series to Dataframe?
    if isinstance(dataframe, pd.Series):
        dataframe = pd.DataFrame(dataframe)

    object_columns_to_category(dataframe)
    columns = (x for x in dataframe.columns.tolist()
               if dataframe[x].dtype != object)
    for column in columns:
        hist_kwargs = {
                    'align': 'mid',
                    }
        serie = dataframe[column].copy(deep=True)
        fig, axes = plt.subplots(nrows=1, ncols=1)
        if normalize:
            weights = np.ones_like(serie)/len(serie)
            hist_kwargs.update({'weights': weights})
        try:
            column_dtype = np.dtype(dataframe[column]).type
        except TypeError:
            column_dtype = type(None)
        # TODO: create a differente procedure for bool.
        if issubclass(column_dtype, np.integer) or issubclass(column_dtype,
                                                              np.bool_):
            # datatype = np.int
            axes = histogram_of_integers(
                dataframe[column],
                **sns_distplot_kwargs)
        elif issubclass(column_dtype, np.float):
            # datatype = np.float
            axes = histogram_of_floats(
                dataframe[column],
                **sns_distplot_kwargs)
        elif issubclass(column_dtype, object):
            if serie.dtype != 'category':
                continue
            axes = histogram_of_integers(
                serie.cat.codes,
                **sns_distplot_kwargs)
            categories_iterator = iter(serie.cat.categories)
            axes.set_xticklabels(
                map(lambda x: categories_iterator.__next__() if x > 0 else '',
                    map(lambda x: x.get_height(),
                        axes.patches)))

        PATCHES_LEN = len(axes.patches)
        PATCHES_STRIDE = 0.2
        FIGSIZE = fig.get_size_inches()
        fig.set_size_inches(
            FIGSIZE[0] + PATCHES_LEN * PATCHES_STRIDE,
            FIGSIZE[1]
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_path,
                '{0}'.format(column)) + '.png',
            dpi=300)
        fig.clf()
        plt.close('all')
    return None


color = {
    'standard': (223, 229, 239),
    'gold': (255, 200, 31),
    'platinum': (192, 192, 192),
    'black': (0, 0, 0),
    'pseudo_black': (90, 90, 90),
    'business': (119, 202, 141),
}
color = {k: (v[0]/255, v[1]/255, v[2]/255) for k, v in color.items()}


if __name__ == '__main__':
    import doctest
    doctest.testmod()
