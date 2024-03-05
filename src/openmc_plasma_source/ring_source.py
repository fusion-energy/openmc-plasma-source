from typing import Tuple

import numpy as np
import openmc

from .fuel_types import get_neutron_energy_distribution


def fusion_ring_source(
    radius: float,
    angles: Tuple[float, float] = (0, 2 * np.pi),
    z_placement: float = 0,
    temperature: float = 20000.0,
    fuel: dict = {"D": 0.5, "T": 0.5},
):
    """An openmc.Source object with some presets to make it more convenient
    for fusion simulations using a ring source. All attributes can be changed
    after initialization if required. Default isotropic ring source with a
    realistic energy distribution.

    Args:
        radius (float): the inner radius of the ring source, in metres
        angles (iterable of floats): the start and stop angles of the ring in radians
        z_placement (float): Location of the ring source (m). Defaults to 0.
        temperature (float): the temperature to use in the Muir distribution in eV,
        fuel (dict): Isotopes as keys and atom fractions as values
    """

    if isinstance(radius, (int, float)) and radius > 0:
        pass
    else:
        raise ValueError("Radius must be a float strictly greater than 0.")

    if (
        isinstance(angles, tuple)
        and len(angles) == 2
        and all(
            isinstance(angle, (int, float)) and -2 * np.pi <= angle <= 2 * np.pi
            for angle in angles
        )
    ):
        pass
    else:
        raise ValueError(
            "Angles must be a tuple of floats between zero and 2 * np.pi"
        )

    if isinstance(z_placement, (int, float)):
        pass
    else:
        raise TypeError("Z placement must be a float.")

    if isinstance(temperature, (int, float)) and temperature > 0:
        pass
    else:
        raise ValueError("Temperature must be a float strictly greater than 0.")

    source = openmc.IndependentSource()

    # performed after the super init as these are Source attributes
    source.space = openmc.stats.CylindricalIndependent(
        r=openmc.stats.Discrete([radius], [1]),
        phi=openmc.stats.Uniform(a=angles[0], b=angles[1]),
        z=openmc.stats.Discrete([z_placement], [1]),
        origin=(0.0, 0.0, 0.0),
    )
    source.angle = openmc.stats.Isotropic()
    source.energy = get_neutron_energy_distribution(
        ion_temperature=temperature, fuel=fuel
    )

    return source
