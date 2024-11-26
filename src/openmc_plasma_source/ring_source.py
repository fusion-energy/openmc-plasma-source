import numpy as np
import openmc.stats
from openmc import IndependentSource

from .fuel_types import get_neutron_energy_distribution

from typing import Tuple, List, Dict


def fusion_ring_source(
    radius: float,
    angles: Tuple[float, float] = (0, 2 * np.pi),
    z_placement: float = 0,
    temperature: float = 20000.0,
    fuel: Dict = {"D": 0.5, "T": 0.5},
) -> List[IndependentSource]:
    """Creates a list of openmc.IndependentSource objects in a ring shape.

    Useful for simulations where all the plasma parameters are not known and
    this simplified geometry will suffice. Resulting ring source will have an
    energy distribution according to the fuel composition.

    Args:
        radius: the inner radius of the ring source, in metres
        angles: the start and stop angles of the ring in
            radians
        z_placement: Location of the ring source (m). Defaults to 0.
        temperature: Temperature of the source (eV).
        fuel: Isotopes as keys and atom fractions as values

    Returns:
        A list of one openmc.IndependentSource instance.
    """

    if not isinstance(radius, (int, float)) or radius <= 0:
        raise ValueError("Radius must be a float strictly greater than 0.")

    if not (
        isinstance(angles, tuple)
        and len(angles) == 2
        and all(
            isinstance(angle, (int, float)) and -2 * np.pi <= angle <= 2 * np.pi
            for angle in angles
        )
    ):
        raise ValueError("Angles must be a tuple of floats between zero and 2 * np.pi")

    if not isinstance(z_placement, (int, float)):
        raise TypeError("Z placement must be a float.")

    if not (isinstance(temperature, (int, float)) and temperature > 0):
        raise ValueError("Temperature must be a float strictly greater than 0.")

    source = IndependentSource()

    source.space = openmc.stats.CylindricalIndependent(
        r=openmc.stats.Discrete([radius], [1]),
        phi=openmc.stats.Uniform(a=angles[0], b=angles[1]),
        z=openmc.stats.Discrete([z_placement], [1]),
        origin=(0.0, 0.0, 0.0),
    )

    energy_distributions = get_neutron_energy_distribution(
        ion_temperature=temperature, fuel=fuel
    )

    source.energy = energy_distributions
    source.angle = openmc.stats.Isotropic()

    return [source]
