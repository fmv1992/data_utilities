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
        (2) Keyword arguments should be turned off by default.

"""

import itertools
import os
import random

import sklearn.preprocessing
from mpl_toolkits.mplot3d import Axes3D
from data_utilities.pandas_utilities import object_columns_to_category
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# pylama: ignore=W0611,D301


def scale_axes_axis(axes, scale_xy_axis=False, scale_z_axis=False):
    """Set axes to the same scale."""
    if hasattr(axes, 'get_zlim'):
        dimenesions = 3
    else:
        dimenesions = 2

    xlim = tuple(axes.get_xlim3d())
    ylim = tuple(axes.get_ylim3d())
    zlim = tuple(axes.get_zlim3d())

    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)

    # Get the maximum absolute difference between the limits in all 3 axis and
    # the mean of the respective axis.
    plot_radius = max(abs(lim - mean_)
                      for lims, mean_ in ((xlim, xmean),
                                          (ylim, ymean),
                                          (zlim, zmean))
                      for lim in lims)

    # Set the span of the axis to be 2 * plot radius for all the plots.
    if scale_xy_axis:
        axes.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
        axes.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    if dimenesions == 3 and scale_z_axis:
        axes.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

    return None


def plot_3d(series,
            colormap_callable=plt.cm.viridis,
            include_colorbar=False):
    """Create a 3d-barchart axes for a given 2-level-multi-index series.

    Return a 3d axes object given a series with a multiindex with 2
    categorical levels.

    Arguments:
        series (pandas.Series): the 2-level-index series to generate the plot.

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
    series = series.copy(deep=True)
    series.sort_values(inplace=True, ascending=False)
    # Set constants.
    # TODO: Make graphs on the same scale effective.
    # View: http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to  # noqa
    SCALE_AXIS_DIST_FACTOR = 1  # This seemingly changes de width of the bar.
    SCALE_AXIS_DIST_TRESH = 1
    # Some groupby objects will produce a dataframe. Not nice going over duck
    # typing but oh well...
    # If it is a dataframe with one column then transform it to series.
    if isinstance(series, pd.DataFrame) and series.shape[1] == 1:
        series = series.ix[:, 0]

    # Error handling phase.
    # Track if index has correct shape.
    if len(series.index.levshape) != 2:
        raise ValueError('The index level shape should '
                         'be 2 and it is {}.'.format(series.index.levshape))
    # Check for duplicate indexes.
    if series.index.duplicated().sum():
        series = series.groupby(level=series.index.names).sum()
        if series.index.duplicated().sum():
            raise ValueError('series has duplicate values.')

    # Handling the index of the received series.
    level1_index, level2_index = tuple(zip(*series.index.get_values()))
    level1_index = sorted(set(level1_index))
    level2_index = sorted(set(level2_index))

    # Populate the series with all index combinations, even if they are zero.
    all_index_combinations = tuple(itertools.product(
        level1_index,
        level2_index))
    index_names = series.index.names
    new_index = pd.MultiIndex.from_tuples(all_index_combinations,
                                          names=index_names)
    all_values_series = pd.Series(0, index=new_index, name=series.name)
    series = (series + all_values_series).fillna(0)

    # Generate the z values
    z_values = []
    for _, group in series.groupby(level=1):
        z_values.append(group)
    z_values = np.hstack(z_values).ravel()
    # TODO: transform all of those into z.
    z = z_values

    # Starts manipulating the axes
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')

    # Create the axis and their labels
    xlabels = series.index.get_level_values(index_names[0]).unique().tolist()
    ylabels = series.index.get_level_values(index_names[1]).unique().tolist()
    xlabels = [''.join(list(filter(str.isalnum, str(value))))
               for value in xlabels]
    ylabels = [''.join(list(filter(str.isalnum, str(value))))
               for value in ylabels]
    x = np.arange(len(xlabels))
    # TODO: Check if this rescaling is working properly.
    if len(x) > SCALE_AXIS_DIST_TRESH:
        x = x * SCALE_AXIS_DIST_FACTOR
    y = np.arange(len(ylabels))
    if len(y) > SCALE_AXIS_DIST_TRESH:
        y = y * SCALE_AXIS_DIST_FACTOR
    xlabels = [z.title() for z in xlabels]
    ylabels = [z.title() for z in ylabels]

    x_mesh, y_mesh = np.meshgrid(x, y, copy=False)

    ax.w_xaxis.set_ticks(x + 0.5/2.)
    ax.w_yaxis.set_ticks(y + 0.5/2.)

    ax.w_xaxis.set_ticklabels(xlabels)
    ax.w_yaxis.set_ticklabels(ylabels)

    # Color.
    pp_color_values = sklearn.preprocessing.minmax_scale(z)
    colormap = colormap_callable(pp_color_values)

    # Create the 3d plot.
    ax.bar3d(x_mesh.ravel(), y_mesh.ravel(), z_values*0,
             dx=0.5, dy=0.5, dz=z_values,
             color=colormap)

    # Set convenient z limits.
    # From z ticks make it include all extreme values in excess of 0.5 tick.
    z_min = z.min()
    z_max = z.max()
    z_ticks = ax.get_zticks()
    z_stride = z_ticks[1] - z_ticks[0]
    z_min_lim = z_min - 0.5 * z_stride
    z_max_lim = z_max + 0.5 * z_stride
    if 0 < z_min_lim:
        z_min_lim = 0
    elif 0 > z_max_lim:
        z_max_lim = 0
    ax.set_zlim3d(z_min_lim, z_max_lim)

    # TODO: allow inclusion of a colorbar.
    # TODO: how to really add ticks not just labels.
    # TODO: add border to colorbar.
    if include_colorbar:
        scalar_mappable = plt.cm.ScalarMappable(cmap=colormap_callable)
        scalar_mappable.set_array([min(z), max(z)])
        scalar_mappable.set_clim(vmin=min(z), vmax=max(z))
        ticks = np.linspace(z_min, z_max, 5)
        colorbar = fig.colorbar(scalar_mappable, drawedges=True, ticks=ticks)
        colorbar.outline.set_visible(True)

    return ax


# TODO: containers.
def label_containers(axes,
                    containers=None,
                    string_formatting=None,
                    label_height_increment=0.01):
    """Attach text labels to axes.

    Arguments:
        axes (matplotlib.axes.Axes): Axes in which text labels will be added.
        containers (list): List of matplotlib.container.Container objects.
        string_fomratting (str): string that will be passed to str.format
        function.

    Returns:
        None

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib_utilities import label_containers
        >>> fig = plt.figure(num=0, figsize=(12, 9))
        >>> ax = fig.add_subplot(1,1,1)
        >>> x = range(10)
        >>> y = range(10)
        >>> ax.bar(x, y)
        <Container object of 10 artists>
        >>> label_containers(ax)
        >>> fig.tight_layout()
        >>> fig.savefig('/tmp/{0}.png'.format('label_containers'))

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

    bar_labels = []
    height_increment = height.max() * label_height_increment
    for i, rect in enumerate(containers):
        label_height = height[i] + height_increment
        text = axes.text(
            rect.get_x() + rect.get_width()/2.,
            label_height,
            string_formatting.format(height[i]),
            ha='center', va='bottom')
        bar_labels.append(text)
    return bar_labels


def histogram_of_categorical(a,
                             *args,
                             **sns_distplot_kwargs):
    """Plot a histogram of categorical with sane defauts.

    Arguments:
        a (pd.Series): Categorical series to create a histogram plot.
    Returns:
        matplotlib.axes.Axes: the plotted axes.

    Examples:
        >>> import pandas_utilities as pu
        >>> cat_serie = pu.dummy_dataframe().categorical
        >>> axes = histogram_of_categorical(cat_serie, kde=False)
        >>> plt.show()

    """
    # Create a dictionary of labels from categories.
    labels = dict(enumerate(a.cat.categories))
    # Pass the arguments to the histogram of integers function.
    axes = histogram_of_integers(a.cat.codes, *args, **sns_distplot_kwargs)
    # Restore the labels.
    new_labels = tuple(map(
        lambda x: labels[x] if x in labels.keys() else '',
        axes.get_xticks()))
    axes.set_xticklabels(new_labels)
    return axes


def histogram_of_floats(a,
                        *args,
                        **sns_distplot_kwargs):
    """Plot a histogram of floats with sane defauts.

    Arguments:
        x (pd.Series): Float series to create a histogram plot.
    Returns:
        matplotlib.axes.Axes: the plotted axes.

    Examples:
        >>> import pandas_utilities as pu
        >>> float_serie = pu.dummy_dataframe().float
        >>> axes = histogram_of_floats(float_serie)
    """
    axes = sns.distplot(
        a,
        *args,
        **sns_distplot_kwargs)
    return axes


def histogram_of_integers(a,
                          *args,
                          **sns_distplot_kwargs):
    """Plot a histogram of integers with sane defauts.

    Arguments:
    Returns:
    Examples:

    """
    # If there are various different integers plot them as float.
    THRESHOLD_TO_CONSIDER_FLOAT = 100
    unique = np.unique(a).shape[0]
    if unique > THRESHOLD_TO_CONSIDER_FLOAT:
        return histogram_of_floats(
            a,
            *args,
            **sns_distplot_kwargs)

    # TODO: Cover the case of less than treshold number of integers but with a
    # lot of spacing between them such as (0, 1, 2, 3, 5500, 15000).
    #
    # An algorithm is needed to find all the numbers that are contiguous and
    # block them in groups. Then split the x axis between those contiguous
    # blocks.
    # TODO: implement such algorithm.
    if a.max() - a.min() > THRESHOLD_TO_CONSIDER_FLOAT:
        unique_values = np.sort(a.unique())
        mask_values = dict(zip(unique_values, range(len(unique_values))))
        a = a.map(mask_values)
    xlabels = np.arange(a.min() - 2,
                        a.max() + 3)

    # Specify default options for histogram.
    if 'hist_kws' not in sns_distplot_kwargs:
        sns_distplot_kwargs['hist_kws'] = dict()
        hist_kws = sns_distplot_kwargs['hist_kws']
    DEFAULT_HIST_KWARGS = {
                    'align': 'mid',
                    'rwidth': 0.5}
    # Update kwargs to matplotlib histogram which were not specified.
    for absent_key in filter(lambda x: x not in
                             hist_kws.keys(),
                             DEFAULT_HIST_KWARGS.keys()):
            hist_kws[absent_key] = DEFAULT_HIST_KWARGS[absent_key]

    xlabels = np.arange(a.min() - 2,
                        a.max() + 3)

    axes = sns.distplot(
        a,
        bins=xlabels - 0.5,
        *args,
        **sns_distplot_kwargs)

    axes.set_xticks(xlabels)
    # If it is the case of having mapped the values.
    try:
        mask_values
        a = a.map({v: k for k, v in mask_values.items()})
        xlabels = np.concatenate((
            np.arange(a.min() - 2, a.min()),
            np.sort(np.unique(a)),
            np.arange(a.max() + 1, a.max() + 3)))
    except NameError:
        pass

    # Apply rotation to labels if they are numerous.
    if max(a) >= 100:
        rotation = -45
    else:
        rotation = 0
    axes.set_xticklabels(xlabels, rotation=rotation)

    return axes


def histogram_of_dataframe(dataframe,
                           output_path=None,
                           normalize=True,
                           weights=None,
                           *args,
                           **sns_distplot_kwargs):
    """Draw a histogram for each column of the dataframe.

    Provide a quick summary of each series in the dataframe:
         - Draw a histogram for each column of the dataframe using the seaborn
         'distplot' function.
         - Create an artist box with some summary statistics:
            - max
            - min
            - average
            - nans
            - n

    The dataframe may contain nans.

    This function assumes that the input dataframe has already received
    treatment such as outlier treatment.

    Arguments:
        dataframe (pandas.DataFrame): The dataframe whose columns will be
        plotted.
        output_folder (str): The outputh path to place the plotted histograms.
        If None then no file is written.
        normalize (bool): If the histograms should normalize so that the bin
        heights add up to 1.
        weights (list): The list of numpy.array weights to weight each of the
        histogram entry.

    Returns:
        tuple: a tuple containing a figure and an axes for each dataframe.

    Examples:
        >>> import pandas_utilities
        >>> dummy_df = pandas_utilities.dummy_dataframe(shape=200)
        >>> histogram_of_dataframe(dummy_df, '/tmp/')

    """
    # This function assumes that the dataframe has received treatment. If there
    # is an object column then raise exceptions. However nan's are welcome as
    # they are part of the informative box.
    if (dataframe.dtypes == object).sum() > 0:
        raise TypeError("Dataframe must not have object columns:\n{0}",
                        dataframe.dtypes)

    list_of_figures = list()

    # Define the standard alignement for the histogram.
    hist_kwargs = {'align': 'mid'}

    # Iterate over columns.
    for i, column in enumerate(dataframe.columns):
        fig, axes = plt.subplots(nrows=1, ncols=1)
        series = dataframe[column]

        # Since numpy dtypes seem not to be organized in a hierarchy of data
        # types (eg int8, int32 etc are instances of a int) we resort to a
        # string representation of data types.
        series_str_dtype = str(series.dtype)

        # TODO: review necessity of this.
        if normalize:
            weights = np.ones_like(series)/len(series)
            hist_kwargs.update({'weights': weights})

        # .
        # ├── categorical (x)
        # └── number
        #     ├── bool
        #     ├── float
        #     └── int
        if series_str_dtype == 'category':
            # Map category to integers and do an integer plot.
            axes = histogram_of_integers(
                series.cat.codes,
                **sns_distplot_kwargs)
            # Revert labels to category names.
            categories_iterator = iter(series.cat.categories)
            axes.set_xticklabels(
                map(lambda x: categories_iterator.__next__() if x > 0 else '',
                    map(lambda x: x.get_height(),
                        axes.patches)))
        # .
        # ├── categorical
        # └── number (x)
        #     ├── bool
        #     ├── float
        #     └── int
        #
        # Series with nans cannot be passed to sns.distplot. So this should be
        # sent separetely to add_summary_statistics_textbox
        elif ('bool' in series_str_dtype or 'int' in series_str_dtype or
              'float' in series_str_dtype):
            # Null values if passed to seaborn.distplot raise ValueError.
            series_not_null = series[~series.isnull()]
            # .
            # ├── categorical
            # └── number
            #     ├── bool (x)
            #     ├── float
            #     └── int
            if 'bool' in series_str_dtype:
                # TODO: implement.
                pass

            # .
            # ├── categorical
            # └── number
            #     ├── bool
            #     ├── float (x)
            #     └── int
            if 'float' in series_str_dtype:
                axes = histogram_of_floats(
                    series_not_null,
                    **sns_distplot_kwargs)

            # .
            # ├── categorical
            # └── number
            #     ├── bool
            #     ├── float
            #     └── int (x)
            if 'int' in series_str_dtype:
                axes = histogram_of_integers(
                    series_not_null,
                    **sns_distplot_kwargs)

            # Add summary statistics for all numeric cases.
            text = add_summary_statistics_textbox(series, axes)  # noqa

        # If it is neither a number nor a categorical data type raise error.
        else:
            raise TypeError("Datatype {0} not covered in this"
                            " function".format(series_str_dtype))

        # Adjust figure size.
        PATCHES_LEN = len(axes.patches)
        PATCHES_STRIDE = 0.2
        FIGSIZE = fig.get_size_inches()
        fig.set_size_inches(
            FIGSIZE[0] + PATCHES_LEN * PATCHES_STRIDE,
            FIGSIZE[1]
        )

        list_of_figures.append(fig)

        # Save the plotting.
        if output_path is not None:
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_path,
                    '{0}'.format(column)) + '.png',
                dpi=300)

    return tuple(list_of_figures)


def add_summary_statistics_textbox(series,
                                   axes,
                                   include_mean=True,
                                   include_max=True,
                                   include_min=True,
                                   include_n=True,
                                   include_nans=True,
                                   include_stddevs=True,
                                   include_stddevp=False):
    """Add a summary statistic textbox to your figure."""
    mean = series.mean()
    summary_max = series.max()
    summary_min = series.min()
    n = series.shape[0]
    nans = series.isnull().sum()
    stddevs = series.std(ddof=1)
    stddevp = series.std(ddof=0)

    # TODO: remove the divide by zero warning.
    metrics = np.fromiter(
        (x for x in
         (mean, summary_max, summary_min, stddevs, stddevp)
         if x != 0),
        dtype=float)
    min_order = np.floor(np.log10(np.abs(metrics))).min()
    if abs(min_order) == float('inf'):
        min_order = 0
    min_order = np.int(min_order)
    expo = 10 ** min_order

    # TODO: numbers and figures should have the same order of magnitude.
    # That is, avoid:
    # mean: 1e7
    # std:  1e5
    # max:  1e9
    # Float.
    text_mean = ('mean = {0:1.2f}' + 'e{1:d}').format(mean/expo, min_order)
    text_max = ('max = {0:1.2f}' + 'e{1:d}').format(summary_max/expo, min_order)
    text_min = ('min = {0:1.2f}' + 'e{1:d}').format(summary_min/expo, min_order)
    text_stddevp = ('stddevp = {0:1.2f}' + 'e{1:d}').format(stddevp/expo,
                                                            min_order)
    # Integers.
    text_n = 'n = {0:d}'.format(n)
    text_nans = 'nans = {0:d} ({1:1.1%} of n)'.format(nans, nans/n)

    text = (text_mean, text_max, text_min, text_n, text_nans, text_stddevp)

    # This session of the code finds the best placement of the text box. It
    # works by finding a sequence of patches that are either above half the y
    # axis or below it. If it finds such a sequences then it places the box
    # halfway of the first patch of this sequence.
    # This minimizes the chances of having it placed in an unsuitable positon.
    n_bins = len(axes.patches)
    stride = axes.patches[0].get_width()
    hist_xlim = (axes.patches[0].get_x(), axes.patches[0].get_x() + n_bins *
                 stride)
    x0 = hist_xlim[0]
    y_half = axes.get_ylim()[1] / 2
    fraction_of_patches_to_halt = 1/4
    contiguous_patches_to_halt = int(n_bins * fraction_of_patches_to_halt)
    patches_height = (x.get_height() for x in axes.patches)
    height_greater_than_half = map(lambda x: x > y_half,
                                   patches_height)
    state = height_greater_than_half.__next__()
    seq = 1
    flipped_on = 1
    for i, greater in enumerate(height_greater_than_half, 1):
        if greater == state:
            seq += 1
        else:
            seq = 1
            state = greater
            flipped_on = i
        if seq >= contiguous_patches_to_halt:
            if greater:
                y_placement = 0.3  # as a fraction of the ylimits.
            else:
                y_placement = 0.95  # as a fraction of the ylimits.
            # Place the box on the best place: half stride in the patch which
            # happened to 'flip' (y_half_greater -> y_half_smaller or vice
            # versa).
            x_placement = ((i - contiguous_patches_to_halt + flipped_on)
                           * stride + x0 + 0.5 * stride)
            break
    else:
        # TODO: implement this scenario.
        raise NotImplementedError("Need to implement the case of not having a "
                                  "suitable sequence of histogram patches.")
    axes_ylim = axes.get_ylim()
    # Correct the placement of the box to absolute units.
    y_placement = axes_ylim[0] + y_placement * (axes_ylim[1] - axes_ylim[0])

    # Set the box style for the text.
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place the text.
    text = axes.text(
        x_placement, y_placement,
        '\n'.join(text),
        verticalalignment='top',
        alpha=0.5,
        bbox=props)

    return text


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
    # doctest.testmod()
    doctest.run_docstring_examples(plot_3d, globals())
