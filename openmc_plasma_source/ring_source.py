import openmc
import numpy as np


class FusionRingSource(openmc.Source):
    """An openmc.Source object with some presets to make it more convenient
    for fusion simulations using a ring source. All attributes can be changed
    after initialization if required. Default isotropic ring source with a Muir
    energy distribution.

    Args:
        radius: the inner radius of the ring source
        angles (iterable of floats): the start and stop angles of the ring in radians,
        z_placement: Location of the ring source (m). Defaults to 0.
        temperature: the temperature to use in the Muir distribution in eV,
    """

    def __init__(
        self,
        radius,
        angles=(0, 2 * np.pi),
        z_placement=0,
        temperature: float = 20000.0,
        fuel="DT",
    ):

        super().__init__()

        # performed after the super init as these are Source attributes
        radius = openmc.stats.Discrete([radius], [1])
        z_values = openmc.stats.Discrete([z_placement], [1])
        angle = openmc.stats.Uniform(a=angles[0], b=angles[1])
        self.space = openmc.stats.CylindricalIndependent(
            r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0)
        )
        self.angle = openmc.stats.Isotropic()
        if fuel == "DT":
            mean_energy = 14080000.0  # mean energy in eV
            mass_of_reactants = 5  # mass of the reactants (D + T) AMU
        elif fuel == "DD":
            mean_energy = 2450000.0  # mean energy in eV
            mass_of_reactants = 4  # mass of the reactants (D + D) AMU
        else:
            raise ValueError(f'fuel must be either "DT" or "DD", not {fuel}')
        self.energy = openmc.stats.Muir(
            e0=mean_energy, m_rat=mass_of_reactants, kt=temperature
        )
