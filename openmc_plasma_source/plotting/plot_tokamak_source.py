import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def scatter_tokamak_source(source, quantity=None, **kwargs):
    """Create a 2D scatter plot of the tokamak source.
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    for more arguments.

    Args:
        source (ops.TokamakSource): the plasma source
        quantity ("str", optional): value by which the lines should be
            coloured. Defaults to None.

    Raises:
        ValueError: If the quantity is unknown
    """

    quantity_to_attribute = {
        "ion_temperature": source.temperatures,
        "neutron_source_density": source.neutron_source_density,
        "strength": source.strengths,
    }
    if quantity in quantity_to_attribute:
        colours = quantity_to_attribute[quantity]
    elif quantity is None:
        colours = None
    else:
        raise ValueError("Unknown quantity")
    plt.gca().set_aspect("equal")

    return plt.scatter(source.RZ[0], source.RZ[1], c=colours, **kwargs)


def plot_tokamak_source_3D(
    source, quantity=None, angles=[0, 1 / 2 * np.pi], colorbar="viridis", **kwargs
):
    """Creates a 3D plot of the tokamak source.
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
    for more arguments.

    Args:
        source (ops.TokamakSource): the plasma source
        quantity ("str", optional): value by which the lines should be
            coloured. Defaults to None.
        angles (list, optional): iterable of two floats defining the coverage.
            Defaults to [0, 1/2*np.pi].
        colorbar (str, optional): colorbar used if quantity is not None.
            Defaults to "viridis".

    Raises:
        ValueError: If the quantity is unknown
    """

    quantity_to_attribute = {
        "ion_temperature": source.temperatures,
        "neutron_source_density": source.neutron_source_density,
        "strength": source.strengths,
    }
    if quantity in quantity_to_attribute:
        values = quantity_to_attribute[quantity]
    elif quantity is None:
        values = None
    else:
        raise ValueError("Unknown quantity")

    colorbar = cm.get_cmap(colorbar)
    axes = plt.axes(projection="3d")
    theta = np.linspace(*angles, 100)
    for i in range(source.sample_size):
        if values is not None:
            colour = colorbar(values[i] / max(values))
        else:
            colour = None
        x = source.RZ[0][i] * np.sin(theta)
        y = source.RZ[0][i] * np.cos(theta)
        z = source.RZ[1][i]
        plt.plot(x, y, z, color=colour, **kwargs)

    axes.set_xlim(-source.major_radius, source.major_radius)
    axes.set_ylim(-source.major_radius, source.major_radius)
    axes.set_zlim(-source.major_radius, source.major_radius)
