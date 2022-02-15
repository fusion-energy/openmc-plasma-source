import openmc
import numpy as np
from typing import Tuple
from param import Parameterized, Number, Range, ListSelector

from .fuel_types import fuel_types


class FusionRingSource(openmc.Source, Parameterized):
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

    radius = Number(None, bounds=(0, None), inclusive_bounds=(False, False))
    angles = Range((0, 2 * np.pi))
    z_placement = Number()
    temperature = Number(bounds=(0, None))
    fuel_type = ListSelector(fuel_types.keys())

    def __init__(
        self,
        radius: float,
        angles: Tuple[float, float] = (0, 2 * np.pi),
        z_placement: float = 0,
        temperature: float = 20000.0,
        fuel: str = "DT",
    ):

        # Set local attributes
        self.radius = radius
        self.angles = angles
        self.z_placement = z_placement
        self.temperature = temperature
        self.fuel_type = fuel
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
