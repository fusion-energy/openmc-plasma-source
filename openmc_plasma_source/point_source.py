
from typing import Tuple

import openmc


class FusionPointSource(openmc.Source):
    """An openmc.Source object with some presets to make it more convenient
    for fusion simulations using a point source. All attributes can be changed
    after initialization if required. Default isotropic point source at the
    origin with a Muir energy distribution. 
    """
    def __init__(
        self,
        coordinate: Tuple[float, float, float] = (0, 0, 0),
        temperature: float = 20000.,
        fuel: str = 'DT'
    ):
 
        super().__init__()

        # performed after the super init as these are Source attributes
        self.space = openmc.stats.Point(coordinate)
        self.angle = openmc.stats.Isotropic()
        if fuel == 'DT':
            mean_energy = 14080000.  # mean energy in eV
            mass_of_reactants = 5  # mass of the reactants (D + T) AMU
        elif fuel == 'DT':
            mean_energy = 2450000.  # mean energy in eV
            mass_of_reactants = 4  # mass of the reactants (D + D) AMU
        else:
            raise ValueError(f'fuel must be either "DT" or "DD", not {fuel}')
        self.energy = openmc.stats.Muir(e0=mean_energy , m_rat=mass_of_reactants , kt=temperature)
