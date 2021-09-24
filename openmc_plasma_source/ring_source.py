
import openmc


class FusionRingSource(openmc.Source):
    """An openmc.Source object with some presets to make it more convenient
    for fusion simulations using a ring source. All attributes can be changed
    after initialization if required. Default isotropic ring source with a Muir
    energy distribution.

    Args:
        radius: the inner radius of the ring source
        start_angle: the start angle of the ring in radians,
        stop_angle: the end angle of the ring in radians,
        temperature: the temperature to use in the Muir distribution in eV,
    """
    def __init__(
        self,
        radius,
        start_angle: float =0.,
        stop_angle: float = 6.28318530718,
        temperature: float = 20000.,
        fuel='DT'
    ):

        super().__init__()

        # performed after the super init as these are Source attributes
        radius = openmc.stats.Discrete([radius], [1])
        z_values = openmc.stats.Discrete([0], [1])
        angle = openmc.stats.Uniform(a=start_angle, b=stop_angle)
        self.space = openmc.stats.CylindricalIndependent(
            r=radius,
            phi=angle,
            z=z_values,
            origin=(0.0, 0.0, 0.0)
        )
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
