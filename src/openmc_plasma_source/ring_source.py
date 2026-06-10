import numpy as np
import openmc.stats
from openmc import IndependentSource

from .fuel_types import get_neutron_energy_distribution

from typing import List, Dict


def fusion_ring_source(
    radius: float,
    start_angle: float = 0.0,
    rotation_angle: float = 2 * np.pi,
    z_placement: float = 0,
    temperature: float = 20000.0,
    fuel: Dict = {"D": 0.5, "T": 0.5},
) -> List[IndependentSource]:
    """Creates a list of openmc.IndependentSource objects in a ring shape.

    Useful for simulations where all the plasma parameters are not known and
    this simplified geometry will suffice. Resulting ring source will have an
    energy distribution according to the fuel composition. The ring spans the
    toroidal sector from ``start_angle`` to ``start_angle + rotation_angle``,
    which guarantees the source cannot overlap with itself.

    Args:
        radius: the inner radius of the ring source, in metres
        start_angle: the toroidal angle at which the ring source starts, in
            radians. Must be between -2 * np.pi and 2 * np.pi.
        rotation_angle: the toroidal extent of the ring source, in radians.
            Must be between -2 * np.pi and 2 * np.pi. A negative value extends
            the source in the opposite direction from start_angle.
        z_placement: Location of the ring source (m). Defaults to 0.
        temperature: Temperature of the source (eV).
        fuel: Isotopes as keys and atom fractions as values

    Returns:
        A list of one openmc.IndependentSource instance.
    """

    if not isinstance(radius, (int, float)) or radius <= 0:
        raise ValueError("Radius must be a float strictly greater than 0.")

    if not (
        isinstance(start_angle, (int, float))
        and not isinstance(start_angle, bool)
        and -2 * np.pi <= start_angle <= 2 * np.pi
    ):
        raise ValueError("start_angle must be a float between -2 * np.pi and 2 * np.pi")

    if not (
        isinstance(rotation_angle, (int, float))
        and not isinstance(rotation_angle, bool)
        and -2 * np.pi <= rotation_angle <= 2 * np.pi
    ):
        raise ValueError(
            "rotation_angle must be a float between -2 * np.pi and 2 * np.pi"
        )

    if not isinstance(z_placement, (int, float)):
        raise TypeError("Z placement must be a float.")

    if not (isinstance(temperature, (int, float)) and temperature > 0):
        raise ValueError("Temperature must be a float strictly greater than 0.")

    source = IndependentSource()

    source.space = openmc.stats.CylindricalIndependent(
        r=openmc.stats.Discrete([radius], [1]),
        phi=openmc.stats.Uniform(a=start_angle, b=start_angle + rotation_angle),
        z=openmc.stats.Discrete([z_placement], [1]),
        origin=(0.0, 0.0, 0.0),
    )

    energy_distributions = get_neutron_energy_distribution(
        ion_temperature=temperature, fuel=fuel
    )

    source.energy = energy_distributions
    source.angle = openmc.stats.Isotropic()

    return [source]
