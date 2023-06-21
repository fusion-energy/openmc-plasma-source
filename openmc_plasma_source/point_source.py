import openmc
from typing import Tuple

from .fuel_types import get_muir_paramters


def fusion_point_source(
    coordinate: Tuple[float, float, float] = (0, 0, 0),
    temperature: float = 20000.0,
    fuel: str = "DT",     
):
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

    source = openmc.IndepedentSource()
    source.space = openmc.stats.Point(coordinate)
    source.angle = openmc.stats.Isotropic()

    mean_energy, mass_of_reactants = get_muir_paramters(fuel)

    source.energy = openmc.stats.muir(
        e0=mean_energy,
        m_rat=mass_of_reactants,
        kt=temperature,
    )

    return source
