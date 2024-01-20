import openmc
from typing import Tuple

from .fuel_types import get_neutron_energy_distribution


class FusionPointSource(openmc.IndependentSource):
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

    def __init__(
        self,
        coordinate: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        temperature: float = 20000.0,
        fuel: dict = {"D": 0.5, "T": 0.5},
    ):
        # Set local attributes
        self.coordinate = coordinate
        self.temperature = temperature
        self.fuel_type = fuel
        self.fuel = fuel_types[self.fuel_type]

        # Call init for openmc.Source
        super().__init__()

        # performed after the super init as these are Source attributes
        self.space = openmc.stats.Point(self.coordinate)
        self.angle = openmc.stats.Isotropic()
        self.energy = get_neutron_energy_distribution(
            ion_temperature=temperature, fuel=fuel
        )

    @property
    def coordinate(self):
        return self._coordinate

    @coordinate.setter
    def coordinate(self, value):
        if (
            isinstance(value, tuple)
            and len(value) == 3
            and all(isinstance(x, (int, float)) for x in value)
        ):
            self._coordinate = value
        else:
            raise ValueError("coordinate must be a tuple of three floats.")

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if isinstance(value, (int, float)) and value > 0:
            self._temperature = value
        else:
            raise ValueError("Temperature must be strictly positive float.")

    @property
    def fuel_type(self):
        return self._fuel_type

    @fuel_type.setter
    def fuel_type(self, value):
        if value in fuel_types:
            self._fuel_type = value
        else:
            raise KeyError("Invalid fuel type")
