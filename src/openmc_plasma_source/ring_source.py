import openmc
import numpy as np
from typing import Tuple

from .fuel_types import fuel_types


class FusionRingSource(openmc.IndependentSource):
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
        self.energy = openmc.stats.muir(
            e0=self.fuel.mean_energy,
            m_rat=self.fuel.mass_of_reactants,
            kt=self.temperature,
        )

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if isinstance(value, (int, float)) and value > 0:
            self._radius = value
        else:
            raise ValueError("Radius must be a float strictly greater than 0.")

    @property
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, value):
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and all(
                isinstance(angle, (int, float)) and -2 * np.pi <= angle <= 2 * np.pi
                for angle in value
            )
        ):
            self._angles = value
        else:
            raise ValueError(
                "Angles must be a tuple of floats between zero and 2 * np.pi"
            )

    @property
    def z_placement(self):
        return self._z_placement

    @z_placement.setter
    def z_placement(self, value):
        if isinstance(value, (int, float)):
            self._z_placement = value
        else:
            raise TypeError("Z placement must be a float.")

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if isinstance(value, (int, float)) and value > 0:
            self._temperature = value
        else:
            raise ValueError("Temperature must be a float strictly greater than 0.")

    @property
    def fuel_type(self):
        return self._fuel_type

    @fuel_type.setter
    def fuel_type(self, value):
        if value in fuel_types.keys():
            self._fuel_type = value
        else:
            raise KeyError("Invalid fuel type.")
