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
import warnings

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # enable scale_axes_axis  # noqa

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.preprocessing

from data_utilities import pandas_utilities as pu


def scale_axes_axis(axes, scale_xy_axis=False, scale_z_axis=False):
    """Set axes to the same scale.

    Arguments:
        axes (matplotlib.axes.Axes): input axes to have its axis scaled.
        scale_xy_axis (bool): True to scale both x and y axis.
        scale_z_axis (bool): True to scale z axis.

    Returns:
        None: axes are scaled inplace.

    Examples:
        >>> import matplotlib_utilities as mu
        >>> from mpl_toolkits.mplot3d import Axes3D
        >>> from mpl_toolkits.mplot3d.art3d import Path3DCollection
        >>> x = np.arange(0, 10)
        >>> y = x.copy()
        >>> xx, yy = np.meshgrid(x, y)
        >>> z = np.log2(xx * yy)
        >>> fig = plt.figure()
        >>> ax = fig.gca(projection='3d')
        >>> isinstance(ax.scatter3D(xx, yy, z), Path3DCollection)
        True
        >>> mu.scale_axes_axis(ax, scale_xy_axis=True, scale_z_axis=True)
        >>> fig.tight_layout()
        >>> fig.savefig('/tmp/doctest_{0}.png'.format('scale_axes_axis'),
        ...             dpi=500)

    """
    if hasattr(axes, 'get_zlim'):
        n_dimensions = 3
    else:
        n_dimensions = 2

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
    if n_dimensions == 3 and scale_z_axis:
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
        colormap_callable (matplotlib.colors.ListedColormap): Colormap object
        generated from a list of colors.
        include_colorbar (bool): True to include a colorbar.

    Returns:
        matplotlib.axes.Axes: the 3d axis object.

    Examples:
        >>> import itertools
        >>> import pandas as pd
        >>> fig = plt.figure()
        >>> s_index = pd.MultiIndex.from_tuples(
        ...     tuple(itertools.product(range(6), list('abc'))),
        ...     names=('x1', 'x2'))
        >>> s = pd.Series(data=np.arange(18), index=s_index)
        >>> ax = plot_3d(s, include_colorbar=True)
        >>> fig.tight_layout()
        >>> fig.savefig('/tmp/{0}.png'.format('plot3d'), dpi=500)

    """
    # Create a copy of the list to avoid changing the original.
    # Set constants.
    # Some groupby objects will produce a dataframe. Not nice going over duck
    # typing but oh well...
    # If it is a dataframe with one column then transform it to series.
    if isinstance(series, pd.DataFrame) and series.shape[1] == 1:
        series = series.ix[:, 0].copy(deep=True)
    else:
        series = series.copy(deep=True)

    series.sort_values(inplace=True, ascending=False)

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
    z_list = []
    for _, group in series.groupby(level=1):
        z_list.append(group)
    z_list = np.hstack(z_list).ravel()
    z = z_list
    del z_list

    # Starts manipulating the axes
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')

    # Create labels.
    ax.set_xlabel(index_names[0])
    ax.set_ylabel(index_names[1])

    # Create the axis and their labels
    xlabels = series.index.get_level_values(index_names[0]).unique().tolist()
    ylabels = series.index.get_level_values(index_names[1]).unique().tolist()
    xlabels = [''.join(list(filter(str.isalnum, str(value))))
               for value in xlabels]
    ylabels = [''.join(list(filter(str.isalnum, str(value))))
               for value in ylabels]
    x = np.arange(len(xlabels))
    y = np.arange(len(ylabels))
    xlabels = [z.title() for z in xlabels]
    ylabels = [z.title() for z in ylabels]
    # Adjust tick posistions and labels.
    ax.w_xaxis.set_ticks(x + 0.5/2.)
    ax.w_yaxis.set_ticks(y + 0.5/2.)
    ax.w_yaxis.set_ticklabels(ylabels)
    ax.w_xaxis.set_ticklabels(xlabels)

    # Color.
    pp_color_values = sklearn.preprocessing.minmax_scale(z)
    colormap = colormap_callable(pp_color_values)

    # Create the 3d plot.
    x_mesh, y_mesh = np.meshgrid(x, y, copy=False)
    ax.bar3d(x_mesh.ravel(), y_mesh.ravel(), z*0,
             dx=0.5, dy=0.5, dz=z,
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

    if include_colorbar:
        scalar_mappable = plt.cm.ScalarMappable(cmap=colormap_callable)
        scalar_mappable.set_array([min(z), max(z)])
        scalar_mappable.set_clim(vmin=min(z), vmax=max(z))
        ticks = np.linspace(z_min, z_max, 5)
        colorbar = fig.colorbar(scalar_mappable, drawedges=True, ticks=ticks)
        # Add border to the colorbar.
        colorbar.outline.set_visible(True)
        colorbar.outline.set_edgecolor('black')
        mpl_params = matplotlib.rc_params()
        colorbar.outline.set_linewidth(mpl_params['lines.linewidth'])
        # Add ticks to the colorbar.
        colorbar.ax.yaxis.set_tick_params(
            width=mpl_params['ytick.major.width'],
            size=mpl_params['ytick.major.size'])
    return ax


def label_containers(axes,
                     containers=None,
                     string_formatting=None,
                     label_height_increment=0.01,
                     adjust_yaxis_limit=True):
    """Attach text labels to axes.

    Arguments:
        axes (matplotlib.axes.Axes): Axes in which text labels will be added.
        containers (list): List of matplotlib.container.Container objects.
        string_fomratting (str): string that will be passed to str.format
        function.
        label_height_increment (float): height to increment the label to avoid
        coincidence with bar's top line.

    Returns:
        list: list of text objects generated by the axes.text method.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib_utilities import label_containers
        >>> fig = plt.figure(num=0, figsize=(12, 9))
        >>> ax = fig.add_subplot(1,1,1)
        >>> x = range(10)
        >>> y = range(10)
        >>> isinstance(ax.bar(x, y), matplotlib.container.Container)
        True
        >>> isinstance(label_containers(ax), list)
        True
        >>> fig.tight_layout()
        >>> fig.savefig('/tmp/{0}.png'.format('label_containers'))

    """
    if containers is None:
        containers = axes.containers[0]
    height = np.fromiter(
        (x.get_height() for x in containers), float)
    if string_formatting is None:
        if np.all(np.equal(np.mod(height, 1), 0)):
            string_formatting = '{0:d}'
            height = height.astype('int')
        else:
            string_formatting = '{0:1.1f}'

    height_increment = height.max() * label_height_increment

    # Adjust y axis limit to avoid text out of chart area.
    if adjust_yaxis_limit:
        y0, y1 = axes.get_ylim()
        y1 += 2 * height_increment
        axes.set_ylim(y0, y1)

    # Plot the labels.
    bar_labels = []
    for i, rect in enumerate(containers):
        label_height = height[i] + height_increment
        text = axes.text(
            rect.get_x() + rect.get_width()/2.,
            label_height,
            string_formatting.format(height[i]),
            ha='center', va='bottom')
        bar_labels.append(text)
    return bar_labels


def histogram_of_categorical(a, *args, **sns_distplot_kwargs):
    """Plot a histogram of categorical with sane defauts.

    Arguments:
        a (pd.Series): Categorical series to create a histogram plot.

    Returns:
        matplotlib.axes.Axes: the plotted axes.

    Examples:
        >>> from data_utilities import pandas_utilities as pu
        >>> cat_serie = pu.dummy_dataframe().categorical_0
        >>> fig = plt.figure()
        >>> axes = histogram_of_categorical(cat_serie, kde=False)
        >>> isinstance(axes, matplotlib.axes.Axes)
        True
        >>> fig.savefig('/tmp/doctest_{0}.png'.format(
        ...     'histogram_of_categorical'), dpi=500)

    """
    axes = sns.countplot(a, *args, **sns_distplot_kwargs)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=-45)
    return axes


def histogram_of_floats(a, *args, **sns_distplot_kwargs):
    """Plot a histogram of floats with sane defauts.

    Arguments:
        a (pd.Series): Float series to create a histogram plot.

    Returns:
        matplotlib.axes.Axes: the plotted axes.

    Examples:
        >>> from data_utilities import pandas_utilities as pu
        >>> float_serie = pu.dummy_dataframe().float_0
        >>> fig = plt.figure()
        >>> axes = histogram_of_floats(float_serie, kde=False)
        >>> isinstance(axes, matplotlib.axes.Axes)
        True
        >>> fig.savefig('/tmp/doctest_{0}.png'.format(
        ...     'histogram_of_floats'), dpi=500)

    """
    axes = sns.distplot(a, *args, **sns_distplot_kwargs)
    return axes


def histogram_of_integers(a, *args, **sns_distplot_kwargs):
    """Plot a histogram of integers with sane defauts.

    Arguments:
        a (pd.Series): Integer series to create a histogram plot.

    Returns:
        matplotlib.axes.Axes: the plotted axes.

    Examples:
        >>> from data_utilities import pandas_utilities as pu
        >>> int_serie = pu.dummy_dataframe().int_0
        >>> fig = plt.figure()
        >>> axes = histogram_of_integers(int_serie, kde=False)
        >>> isinstance(axes, matplotlib.axes.Axes)
        True
        >>> fig.savefig('/tmp/doctest_{0}.png'.format(
        ...     'histogram_of_ints'), dpi=500)

    """
    # Data transformation:
    if not isinstance(a, pd.Series):
        a = pd.Series(a)

    # If there are various different integers plot them as float.
    THRESHOLD_TO_CONSIDER_FLOAT = 100
    unique = np.unique(a).shape[0]
    if unique > THRESHOLD_TO_CONSIDER_FLOAT:
        return histogram_of_floats(
            a,
            *args,
            **sns_distplot_kwargs)

    # Mask values if the range between maximum and minimum is too big.
    if a.max() - a.min() > THRESHOLD_TO_CONSIDER_FLOAT:
        unique_values = np.sort(a.unique())
        mask_values = dict(zip(unique_values, range(len(unique_values))))
        a = a.map(mask_values)

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
    # TODO: rotate categorical names if they are too many.
    if max(a) >= 100:
        rotation = -45
    else:
        rotation = 0
    axes.set_xticklabels(xlabels, rotation=rotation)

    return axes


def histogram_of_dataframe(dataframe,
                           output_path=None,
                           *args,
                           sns_distplot_kwargs=None,
                           sns_countplot_kwargs=None):
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

    Requirements for the dataframe:
        - The dataframe may contain nans.
        - Columns can be either numerical or non numerical.

    Arguments:
        dataframe (pandas.DataFrame): The dataframe whose columns will be
        plotted.
        output_path (str): The outputh path to place the plotted histograms.
        If None then no file is written.

    Returns: None.

    Examples:
        >>> from data_utilities import pandas_utilities as pu
        >>> from data_utilities import matplotlib_utilities as mu
        >>> dummy_df = pu.dummy_dataframe(shape=200)
        >>> df_columns = tuple(x for x in dummy_df.columns if 'object_'
        ...                    not in x)
        >>> dummy_df = dummy_df.loc[:, df_columns]
        >>> isinstance(mu.histogram_of_dataframe(dummy_df, '/tmp/'), tuple)
        True

    """
    if sns_distplot_kwargs is None:
        sns_distplot_kwargs = dict()
    if sns_countplot_kwargs is None:
        sns_countplot_kwargs = dict()
    # Iterate over columns.
    for i, column in enumerate(dataframe.columns):
        fig, axes = plt.subplots(nrows=1, ncols=1)
        series = dataframe[column]

        # Since numpy dtypes seem not to be organized in a hierarchy of data
        # types (eg int8, int32 etc are instances of a int) we resort to a
        # string representation of data types.
        series_str_dtype = str(series.dtypes)

        # Create an axes object for each case.
        if series_str_dtype == 'category' or series_str_dtype == 'object':
            axes = histogram_of_categorical(series, **sns_countplot_kwargs)
        elif ('bool' in series_str_dtype or 'int' in series_str_dtype or
              'float' in series_str_dtype or 'datetime' in series_str_dtype):
            # Null values if passed to seaborn.distplot raise ValueError.
            series_not_null = series[~series.isnull()]
            if series_not_null.empty:
                warnings.warn(
                    'Series with column {0} is empty. Skipping.'.format(
                        column),
                    UserWarning)
                continue
            elif 'bool' in series_str_dtype:
                axes = histogram_of_categorical(
                    series_not_null.astype('category'),
                    **sns_countplot_kwargs)
            elif 'float' in series_str_dtype:
                axes = histogram_of_floats(series_not_null,
                                           **sns_distplot_kwargs)
            elif 'datetime' in series_str_dtype:
                series_not_null = pd.to_numeric(series_not_null)
                # ??? Fix datetime treatment here.
                series = pd.to_numeric(series)
                axes = histogram_of_floats(series_not_null,
                                           **sns_distplot_kwargs)
            elif 'int' in series_str_dtype:
                axes = histogram_of_integers(series_not_null,
                                             **sns_distplot_kwargs)
            else:
                raise NotImplementedError
        # If it is neither a number nor a categorical data type raise error.
        else:
            raise TypeError("Datatype {0} not covered in this"
                            " function".format(series_str_dtype))

        # Add summary statistics for all numeric cases.
        # ??? Reimplement.

        # Adjust figure size.
        PATCHES_LEN = len(axes.patches)
        PATCHES_STRIDE = 0.2
        FIGSIZE = fig.get_size_inches()
        fig.set_size_inches(
            FIGSIZE[0] + PATCHES_LEN * PATCHES_STRIDE,
            FIGSIZE[1]
        )

        # TODO: add a low_memory argument to prevent figures using too much
        # memory.

        # Save the plotting.
        if output_path is not None:
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_path,
                    '{0}'.format(column)) + '.png',
                dpi=300)
            plt.close(fig)

    return None


def change_axis_xticklabels(axis, dict_or_callable):
    """Change axis xticklabels inplace."""
    # TODO: Infer data type of dict.
    dict_values_dtype = type(tuple(dict_or_callable.keys())[0])
    # TODO: change from text to appropriate data type.
    # x1 = axis.get_ticklabels()[0].get_text()
    # import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
    axis.set_ticklabels(
        map(lambda x: dict_or_callable[dict_values_dtype(float(x.get_text()))],
            axis.get_ticklabels()))


def add_summary_statistics_textbox(series,
                                   axes,
                                   include_mean=True,
                                   include_max=True,
                                   include_min=True,
                                   include_n=True,
                                   include_nans=True,
                                   include_stddevs=True,
                                   include_stddevp=False):
    """Add a summary statistic textbox to your figure.

    Arguments:
        series (pd.Series): Series to have summary statistics computed on.
        axes (matplotlib.axes.Axes): axes which will receive the text.
        various inclues (bool): To include or not to include various summary
        statistics.

    Returns:
        matplotlib.text.Text: The drawed text object.

    Examples:
        >>> from data_utilities import pandas_utilities as pu
        >>> serie = pu.dummy_dataframe().int_0
        >>> fig = plt.figure()
        >>> axes = histogram_of_integers(serie, kde=False)
        >>> text = add_summary_statistics_textbox(serie, axes)
        >>> fig.savefig('/tmp/doctest_{0}.png'.format(
        ...             'add_summary_statistics_textbox'), dpi=500)

    """
    def find_best_placement_for_summary_in_histogram(axes):
        """Find the best placement for summary in histogram.

        Arguments:
            axes (matplotlib.axes.Axes): histogram axes with the patches
            properties.

        Returns:
            tuple: A tuple with the (x, y) coordinates for box placement.

        """
        # Find best position for the text box.
        #
        # This session of the code finds the best placement of the text box. It
        # works by finding a sequence of patches that are either above half the
        # y axis or below it. If it finds such a sequences then it places the
        # box halfway of the first patch of this sequence. This minimizes the
        # chances of having it placed in an unsuitable positon.
        n_bins = len(axes.patches)
        stride = axes.patches[0].get_width()
        hist_xlim = (axes.patches[0].get_x(), axes.patches[0].get_x() + n_bins
                     * stride)
        x0 = hist_xlim[0]
        y_half = axes.get_ylim()[1] / 2
        fraction_of_patches_to_halt = 1/4
        contiguous_patches_to_halt = int(n_bins * fraction_of_patches_to_halt)
        patches_height = (x.get_height() for x in axes.patches)
        height_greater_than_half = map(lambda x: x > y_half, patches_height)
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
                # Place the box on the best place: half stride in the patch
                # which happened to 'flip' (y_half_greater -> y_half_smaller or
                # vice versa).
                x_placement = ((i - contiguous_patches_to_halt + flipped_on) *
                               stride + x0 + 0.5 * stride)
                break
        else:
            # TODO: not elegant at all.
            try:
                x_placement
            except NameError:
                x_placement = 0.05
            try:
                y_placement
            except NameError:
                y_placement = 0.95
        axes_ylim = axes.get_ylim()
        # Correct the placement of the box to absolute units.
        y_placement = axes_ylim[0] + y_placement * (axes_ylim[1] -
                                                    axes_ylim[0])

        return (x_placement, y_placement)

    if not isinstance(axes, matplotlib.axes.Axes):
        axes = plt.gca()

    mean = series.mean()
    summary_max = series.max()
    summary_min = series.min()
    n = series.shape[0]
    nans = series.isnull().sum()
    stddevs = series.std(ddof=1)
    stddevp = series.std(ddof=0)

    # Numbers and figures should have the same order of magnitude.
    # That is, avoid:
    # mean: 1e7
    # std:  1e5
    # max:  1e9
    metrics = np.fromiter(
        (x for x in
         (mean, summary_max, summary_min, stddevs, stddevp)
         if x != 0),  # Removes the divide by zero warning.
        dtype=float)
    if len(metrics) != 0:
        min_order = np.floor(np.log10(np.abs(metrics))).min()
        if abs(min_order) == float('inf'):
            min_order = 0
    else:
        min_order = 0
    min_order = np.int(min_order)
    expo = 10 ** min_order

    # Float.
    text_mean = ('mean = {0:1.2f}' + 'e{1:d}').format(mean/expo, min_order)
    text_max = ('max = {0:1.2f}' + 'e{1:d}').format(summary_max/expo,
                                                    min_order)
    text_min = ('min = {0:1.2f}' + 'e{1:d}').format(summary_min/expo,
                                                    min_order)
    text_stddevp = ('stddevp = {0:1.2f}' + 'e{1:d}').format(stddevp/expo,
                                                            min_order)
    # Integers.
    text_n = 'n = {0:d}'.format(n)
    text_nans = 'nans = {0:d} ({1:1.1%} of n)'.format(nans, nans/n)

    text = (text_mean, text_max, text_min, text_n, text_nans, text_stddevp)

    if axes.patches:
        x_placement, y_placement = (
            find_best_placement_for_summary_in_histogram(axes))
    else:
        offset = .1
        x_placement, y_placement = offset, 1 - offset

    # Set the box style for the text.
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place the text.
    text = axes.text(
        x_placement, y_placement,
        '\n'.join(text),
        verticalalignment='top',
        alpha=0.5,
        transform=axes.transAxes,  # make it relative to axes coords.
        bbox=props)

    return text


def list_usable_backends():
    """List all usable backends for matplotlib.

    Arguments:
        (empty)

    Returns:
        list: a list of usable backends for current environment.

    Examples:
        >>> import matplotlib_utilities as mu
        >>> available_backends = mu.list_usable_backends()
        >>> 'agg' in available_backends
        True

    """
    backend_string = ("import matplotlib; matplotlib.use(\"{0}\");"
                      "import matplotlib.pyplot as plt")

    command_string = 'python3 -c \'{0}\' 2>/dev/null'

    usable_backends = []
    for backend in matplotlib.rcsetup.all_backends:
        backend_call = backend_string.format(backend)
        command_call = command_string.format(backend_call)
        return_value = os.system(command_call)
        if return_value == 0:
            usable_backends.append(backend)

    return usable_backends


if __name__ == '__main__':
    import doctest
    doctest.testmod()
