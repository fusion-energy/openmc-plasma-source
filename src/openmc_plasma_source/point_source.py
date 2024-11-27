from typing import Tuple

import openmc.stats
from openmc import IndependentSource

from .fuel_types import get_neutron_energy_distribution

from typing import Dict, List


def fusion_point_source(
    coordinate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    temperature: float = 20000.0,
    fuel: Dict[str, float] = {"D": 0.5, "T": 0.5},
) -> List[IndependentSource]:
    """
    Creates a list of openmc.IndependentSource objects representing a point source.
    Example uses: sealed-tube neutron generator, inertial confinement fusion source...

    Resulting source will have an energy distribution according to the fuel composition.

    Args:
        coordinate: Location of the point source.
            Each component is measured in metres.
        temperature: Temperature of the source (eV).
        fuel: Isotopes as keys and atom fractions as values

    Returns:
        A list of one openmc.IndependentSource instance.
    """

    if not (
        isinstance(coordinate, tuple)
        and len(coordinate) == 3
        and all(isinstance(x, (int, float)) for x in coordinate)
    ):
        raise ValueError("coordinate must be a tuple of three floats.")

    if not isinstance(temperature, (int, float)):
        raise ValueError("Temperature must be a float.")
    if temperature <= 0:
        raise ValueError("Temperature must be positive float.")

    source = openmc.IndependentSource()

    energy_distribution = get_neutron_energy_distribution(
        ion_temperature=temperature, fuel=fuel
    )
    source.energy = energy_distribution
    source.space = openmc.stats.Point(coordinate)
    source.angle = openmc.stats.Isotropic()

    return [source]
