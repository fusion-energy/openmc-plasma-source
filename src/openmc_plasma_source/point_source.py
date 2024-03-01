import openmc
from typing import Tuple

from .fuel_types import get_neutron_energy_distribution


def fusion_point_source(
    coordinate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    temperature: float = 20000.0,
    fuel: dict = {"D": 0.5, "T": 0.5},
):
    """An openmc.Source object with some presets to make it more convenient
    for fusion simulations using a point source. All attributes can be changed
    after initialization if required. Default isotropic point source at the
    origin with a Muir energy distribution.

    Args:
        coordinate (tuple[float,float,float]): Location of the point source.
            Each component is measured in metres.
        temperature (float): Temperature of the source (eV).
        fuel (dict): Isotopes as keys and atom fractions as values
    """

    if (
        isinstance(coordinate, tuple)
        and len(coordinate) == 3
        and all(isinstance(x, (int, float)) for x in coordinate)
    ):
        pass
    else:
        raise ValueError("coordinate must be a tuple of three floats.")

    if not isinstance(temperature, (int, float)):
        raise ValueError("Temperature must be a float.")
    if temperature <= 0:
        raise ValueError("Temperature must be positive float.")

    sources = []

    energy_distributions, strengths = get_neutron_energy_distribution(
        ion_temperature=temperature, fuel=fuel
    )

    # if isinstance(energy_distributions, openmc.stats.Normal) or isinstance(energy_distributions, openmc.stats.Discrete) or isinstance(energy_distributions, openmc.stats.Tabular):
    #     source = openmc.Source()
    #     source.energy = energy_distributions
    #     source.space = openmc.stats.Point(coordinate)
    #     source.angle = openmc.stats.Isotropic()
    #     return source

    # else:
    for energy_distribution, strength in zip(energy_distributions, strengths):
        source = openmc.Source()
        source.energy = energy_distribution
        source.space = openmc.stats.Point(coordinate)
        source.angle = openmc.stats.Isotropic()
        source.strength = strength
        sources.append(source)

    return sources
