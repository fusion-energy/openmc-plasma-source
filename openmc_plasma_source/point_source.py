import openmc
from typing import Tuple
from param import Parameterized, Number, NumericTuple, ListSelector

from .fuel_types import fuel_types


class FusionPointSource(openmc.Source, Parameterized):
    """An openmc.Source object with some presets to make it more convenient
    for fusion simulations using a point source. All attributes can be changed
    after initialization if required. Default isotropic point source at the
    origin with a Muir energy distribution.

    Args:
        coordinate (tuple[float,float,float]): Location of the point source.
            Each component is measured in metres.
        temperature (float): Temperature of the source (eV).
        fuel_type (str): The fusion fuel mix. Either 'DT' or 'DD'.
    """

    coordinate = NumericTuple(None, length=3)
    temperature = Number(None, bounds=(0, None))  # temperature in eV
    fuel_type = ListSelector(fuel_types.keys())

    def __init__(
        self,
        coordinate: Tuple[float, float, float] = (0, 0, 0),
        temperature: float = 20000.0,
        fuel: str = "DT",
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
        self.energy = openmc.stats.Muir(
            e0=self.fuel.mean_energy,
            m_rat=self.fuel.mass_of_reactants,
            kt=self.temperature,
        )
