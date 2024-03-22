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
) -> list[openmc.IndependentSource]:
    """Creates a list of openmc.IndependentSource objects in a ring shape.

    Useful for simulations where all the plasma parameters are not known and
    this simplified geometry will suffice. Resulting ring source will have an
    energy distribution according to the fuel composition.

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
        raise ValueError("Angles must be a tuple of floats between zero and 2 * np.pi")

    if isinstance(z_placement, (int, float)):
        pass
    else:
        raise TypeError("Z placement must be a float.")

    if isinstance(temperature, (int, float)) and temperature > 0:
        pass
    else:
        raise ValueError("Temperature must be a float strictly greater than 0.")

    sources = []

    energy_distributions, strengths = get_neutron_energy_distribution(
        ion_temperature=temperature, fuel=fuel
    )

    for energy_distribution, strength in zip(energy_distributions, strengths):
        source = openmc.IndependentSource()

        source.space = openmc.stats.CylindricalIndependent(
            r=openmc.stats.Discrete([radius], [1]),
            phi=openmc.stats.Uniform(a=angles[0], b=angles[1]),
            z=openmc.stats.Discrete([z_placement], [1]),
            origin=(0.0, 0.0, 0.0),
        )

        source.energy = energy_distribution
        source.angle = openmc.stats.Isotropic()
        source.strength = strength
        sources.append(source)

    return sources
