from typing import Tuple
import numpy as np
import proper_tea as pt
import proper_tea.numpy
from .fuel_types import fuel_types

import openmc


class FusionRingSource(openmc.Source):
    """An openmc.Source object with some presets to make it more convenient
    for fusion simulations using a ring source. All attributes can be changed
    after initialization if required. Default isotropic ring source with a Muir
    energy distribution.

    Args:
        radius (float): the inner radius of the ring source, in metres
        angles (iterable of floats): the start and stop angles of the ring in radians
        z_placement (float): Location of the ring source (m). Defaults to 0.
        temperature (float): the temperature to use in the Muir distribution in eV,
        fuel_type (str): The fusion fuel mix. Either 'DT' or 'DD'.
    """

    radius = pt.positive_float()
    angles = pt.numpy.numpy_array(shape=(2,), dtype=float, sort=True)
    z_placement = pt.floating_point()
    temperature = pt.positive_float()
    fuel_type = pt.in_set(fuel_types.keys())

    def __init__(
        self,
        radius: float,
        angles: Tuple[float, float] = (0, 2 * np.pi),
        z_placement: float = 0,
        temperature: float = 20000.0,
        fuel_type: str = "DT",
    ):

        # Set local attributes
        self.radius = radius
        self.angles = angles
        self.z_placement = z_placement
        self.temperature = temperature
        self.fuel_type = fuel_type
        self.fuel = fuel_types[self.fuel_type]

        # Call init for openmc.Source
        super().__init__()

        # performed after the super init as these are Source attributes
        self.space = openmc.stats.CylindricalIndependent(
            r=openmc.stats.Discrete([self.radius], [1]),
            phi=openmc.stats.Uniform(a=self.angles[0], b=self.angles[1]),
            z=openmc.stats.Discrete([self.z_placement], [1]),
            origin=(0.0, 0.0, 0.0),
        )
        self.angle = openmc.stats.Isotropic()
        self.energy = openmc.stats.Muir(
            e0=self.fuel.mean_energy,
            m_rat=self.fuel.mass_of_reactants,
            kt=self.temperature,
        )
