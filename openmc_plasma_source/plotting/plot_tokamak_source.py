import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def scatter_tokamak_source(source, ax=None, quantity=None, aspect="equal", **kwargs):
    """Create a 2D scatter plot of the tokamak source.
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    for more arguments.

    Args:
        source (ops.TokamakSource): the plasma source
        ax (maplotlib.Axes, optional): Axes object on which to plot. If not
            provided by the user, the current working Axes is retrieved using
            matplotlib.pyplot.gca().
        aspect (str, optional): Set aspect ratio of the plot. May be set to 'auto',
            'equal', or a number denoting the ratio of the plot height to the plot
            width.
        quantity (str, optional): value by which the lines should be
            coloured. Defaults to None.
        **kwargs: Keyword arguments compatible with matplotlib.pyplot.scatter

    Raises:
        ValueError: If the quantity is unknown
    """

    # Define possible quantities, and link to arrays within tokamak_source
    # If given a non-TokamakSource, this step will fail with an AttributeError
    try:
        quantity_to_attribute = {
            "ion_temperature": source.temperatures,
            "neutron_source_density": source.neutron_source_density,
            "strength": source.strengths,
        }
    except AttributeError as e:
        raise ValueError(
            f"openmc_plasma_source.scatter_tokamak_source: argument 'source' "
            f"must be of type TokamakSource"
        ) from e

    # For a given quantity, determine colours to plot for each point
    # If given an incorrect quantity name, this step will fail with a KeyError
    colours = None
    if quantity is not None:
        try:
            colours = quantity_to_attribute[quantity]
        except KeyError as e:
            raise ValueError(
                f"openmc_plasma_source.scatter_tokamak_source: Unknown 'quantity' "
                f"provided, options are {quantity_to_attribute.keys()}"
            ) from e

    # If not provided with an Axes instance, retrieve the current Axes in focus
    if ax is None:
        ax = plt.gca()

    # Scatter the source R and Z positions, optionally colouring using the chosen
    # quantity.
    ax.scatter(source.RZ[0], source.RZ[1], c=colours, **kwargs)

    # Set the aspect ratio on the axes.
    # Defaults to 'equal', so 1m on the x-axis has the same width as 1m on the y-axis
    ax.set_aspect(aspect)

    return ax


def plot_tokamak_source_3D(
    tokamak_source,
    ax=None,
    quantity=None,
    angles=[0, 0.5 * np.pi],
    colorbar=None,
    **kwargs,
):
    """Creates a 3D plot of the tokamak source.
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
    for more arguments.

    Args:
        tokamak_source (ops.TokamakSource): the plasma source
        ax (maplotlib.Axes, optional): Axes object on which to plot. If not
            provided by the user, a new Axes is created by calling
            matplotlib.pyplot.axes(projection="3d").
        quantity ("str", optional): value by which the lines should be
            coloured. Defaults to None.
        angles (list, optional): iterable of two floats defining the coverage.
            Defaults to [0, 1/2*np.pi].
        colorbar (str, optional): colorbar used if quantity is not None.
            When None, matplotlib currently defaults to "viridis".

    Raises:
        ValueError: If the quantity is unknown
    """

    # Define possible quantities, and link to arrays within tokamak_source
    # If given a non-TokamakSource, this step will fail with an AttributeError
    try:
        quantity_to_attribute = {
            "ion_temperature": tokamak_source.temperatures,
            "neutron_source_density": tokamak_source.neutron_source_density,
            "strength": tokamak_source.strengths,
        }
    except AttributeError as e:
        raise ValueError(
            f"openmc_plasma_source.plot_tokamak_source_3D: argument 'source' "
            f"must be of type TokamakSource"
        ) from e

    # For a given quantity, get the corresponding array from tokamak_source
    # If given an incorrect quantity name, this step will fail with a KeyError
    if quantity is not None:
        try:
            quantity_values = quantity_to_attribute[quantity]
        except KeyError as e:
            raise ValueError(
                f"openmc_plasma_source.plot_tokamak_source_3D: Unknown 'quantity' "
                f"provided, options are {quantity_to_attribute.keys()}"
            ) from e
    else:
        quantity_values = np.ones(tokamak_source.sample_size)

    # Get the colours used to plot each curve
    # If 'quantity' == None, all have the same colour, selected from the middle
    # of the colormap.
    cmap = cm.get_cmap(colorbar)
    if quantity is not None:
        colors = cmap(quantity_values / np.max(quantity_values))
    else:
        colors = cmap(np.full(tokamak_source.sample_size, 0.5))

    # If not provided with an Axes object, create a new one
    if ax is None:
        ax = plt.axes(projection="3d")

    # Get curves to plot
    n_theta = 100
    theta = np.linspace(*angles, n_theta)
    xs = np.outer(tokamak_source.RZ[0], np.sin(theta))
    ys = np.outer(tokamak_source.RZ[0], np.cos(theta))
    zs = tokamak_source.RZ[1]

    # Plot each curve in turn
    for x, y, z, color in zip(xs, ys, zs, colors):
        ax.plot(x, y, z, color=color, **kwargs)

    # Set plot bounds
    major_radius = tokamak_source.major_radius
    ax.set_xlim(-major_radius, major_radius)
    ax.set_ylim(-major_radius, major_radius)
    ax.set_zlim(-major_radius, major_radius)

    return ax
