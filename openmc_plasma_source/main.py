import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.lib.function_base import iterable


class Plasma():
    """Plasma neutron source sampling.
    This class greatly relies on models described in [1]

    [1] : Fausser et al, 'Tokamak D-T neutron source models for different
    plasma physics confinement modes', Fus. Eng. and Design,
    https://doi.org/10.1016/j.fusengdes.2012.02.025

    Usage:
        my_plasma = Plasma(**plasma_prms)
        my_plasma.sample_sources()
        print(my_plasma.RZ)
        print(my_plasma.temperatures)
        openmc_sources = my_plasma.make_openmc_sources()

    Args:
        major_radius (float): Plasma major radius (m)
        minor_radius (float): Plasma minor radius (m)
        elongation (float): Plasma elongation
        triangularity (float): Plasma triangularity
        mode (str): Confinement mode ("L", "H", "A")
        ion_density_centre (float): Ion density at the plasma centre (m-3)
        ion_density_peaking_factor (float): Ion density peaking factor
            (refered in [1] as ion density exponent)
        ion_density_pedestal (float): Ion density at pedestal (m-3)
        ion_density_separatrix (float): Ion density at separatrix (m-3)
        ion_temperature_centre (float): Ion temperature at the plasma
            centre (keV)
        ion_temperature_peaking_factor (float): Ion temperature peaking
            factor (refered in [1] as ion temperature exponent alpha_T)
        ion_temperature_beta (float): Ion temperature beta exponent
            (refered in [1] as ion temperature exponent beta_T)
        ion_temperature_pedestal (float): Ion temperature at pedestal (keV)
        ion_temperature_separatrix (float): Ion temperature at separatrix
            (keV)
        pedestal_radius (float): Minor radius at pedestal (m)
        shafranov_factor (float): Shafranov factor (refered in [1] as esh)
            also known as outward radial displacement of magnetic surfaces
            (m)
        sample_size (int, optional): number of neutron sources. Defaults
            to 1000.
    """
    def __init__(
        self,
        major_radius,
        minor_radius,
        elongation,
        triangularity,
        mode,
        ion_density_centre,
        ion_density_peaking_factor,
        ion_density_pedestal,
        ion_density_separatrix,
        ion_temperature_centre,
        ion_temperature_peaking_factor,
        ion_temperature_beta,
        ion_temperature_pedestal,
        ion_temperature_separatrix,
        pedestal_radius,
        shafranov_factor,
        sample_size=1000
    ) -> None:
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.elongation = elongation
        self.triangularity = triangularity
        self.ion_density_centre = ion_density_centre
        self.mode = mode
        self.ion_density_peaking_factor = ion_density_peaking_factor
        self.pedestal_radius = pedestal_radius
        self.ion_density_pedestal = ion_density_pedestal
        self.ion_density_separatrix = ion_density_separatrix

        self.ion_temperature_centre = ion_temperature_centre
        self.ion_temperature_peaking_factor = ion_temperature_peaking_factor
        self.ion_temperature_pedestal = ion_temperature_pedestal
        self.ion_temperature_separatrix = ion_temperature_separatrix
        self.ion_temperature_beta = ion_temperature_beta

        self.shafranov_factor = shafranov_factor

        self.sample_size = sample_size

    def ion_density(self, r):
        """Computes the ion density at a given position. The ion density is
        only dependent on the minor radius.

        Args:
            r (float, ndarray): the minor radius (m)

        Returns:
            float, ndarray: ion density in m-3
        """
        if self.mode == "L":
            density = self.ion_density_centre * \
                (1 - (r/self.major_radius)**2)**self.ion_density_peaking_factor
        elif self.mode in ["H", "A"]:
            # TODO: find an alternative to iterating through the array
            if isinstance(r, np.ndarray):
                density = []
                for radius in r:
                    if 0 < radius < self.pedestal_radius:
                        prod = self.ion_density_centre - self.ion_density_pedestal
                        prod *= (1 - (radius/self.pedestal_radius)**2)**self.ion_density_peaking_factor

                        density_loc = self.ion_density_pedestal + prod
                    else:
                        prod = self.ion_density_pedestal - self.ion_density_separatrix
                        prod *= (self.major_radius - radius)/(self.major_radius - self.pedestal_radius)
                        density_loc = self.ion_density_separatrix + prod
                    density.append(density_loc)
                density = np.array(density)
        return density

    def ion_temperature(self, r):
        """Computes the ion temperature at a given position. The ion
        temperature is only dependent on the minor radius.

        Args:
            r (float, ndarray): minor radius (m)

        Returns:
            float, ndarray: ion temperature (keV)
        """
        if self.mode == "L":
            temperature = self.ion_temperature_centre * \
                (1 - (r/self.major_radius)**2)**self.ion_temperature_peaking_factor
        elif self.mode in ["H", "A"]:
            # TODO: find an alternative to iterating through the array
            if isinstance(r, np.ndarray):
                temperature = []
                for radius in r:
                    if 0 < radius < self.pedestal_radius:
                        prod = self.ion_temperature_centre - self.ion_temperature_pedestal
                        prod *= (1 - (radius/self.pedestal_radius)**self.ion_temperature_beta)**self.ion_temperature_peaking_factor

                        temperature_loc = self.ion_temperature_pedestal + prod
                    else:
                        prod = self.ion_temperature_pedestal - self.ion_temperature_separatrix
                        prod *= (self.major_radius - radius)/(self.major_radius - self.pedestal_radius)
                        temperature_loc = self.ion_temperature_separatrix + prod
                    temperature.append(temperature_loc)
                temperature = np.array(temperature)
        return temperature

    def convert_a_alpha_to_R_Z(self, a, alpha):
        """Converts (r, alpha) cylindrical coordinates to (R, Z) cartesian
        coordinates.

        Args:
            a (float, ndarray): minor radius (m)
            alpha (float, ndarray): angle (rad)

        Returns:
            ((float, ndarray), (float, ndarray)): (R, Z) coordinates
        """
        shafranov_shift = self.shafranov_factor*(1.0-(a/self.minor_radius)**2)
        R = self.major_radius + \
            a*np.cos(alpha + (self.triangularity*np.sin(alpha))) + \
            shafranov_shift
        Z = self.elongation*a*np.sin(alpha)
        return (R, Z)

    def sample_sources(self):
        """Samples self.sample_size neutrons and creates attributes .densities
            (ion density), .temperatures (ion temperature), .strengths
            (neutron source density) and .RZ (coordinates)
        """
        # create a sample of (a, alpha) coordinates
        a = np.random.random(self.sample_size)*self.minor_radius
        alpha = np.random.random(self.sample_size)*2*np.pi

        # compute densities, temperatures, neutron source densities and
        # convert coordinates
        self.densities = self.ion_density(a)
        self.temperatures = self.ion_temperature(a)
        self.strengths = strength(self.densities, self.temperatures)
        self.RZ = self.convert_a_alpha_to_R_Z(a, alpha)

    def make_openmc_sources(self, angles=(0., 2*np.pi)):
        """Creates a list of OpenMC Sources() objects. The created sources are
        ring sources based on the .RZ coordinates between two angles. The
        energy of the sources are Muir energy spectra with ion temperatures
        based on .temperatures. The strength of the sources (their probability)
        is based on .strengths.

        Args:
            angles ((float, float), optional): rotation of the ring source.
            Defaults to (0, 2*np.pi).

        Returns:
            list: list of openmc.Source()
        """
        import openmc

        sources = []
        # create a ring source for each sample in the plasma source
        for i in range(len(self.strengths)):
            my_source = openmc.Source()

            # extract the RZ values accordingly
            radius = openmc.stats.Discrete([self.RZ[0][i]], [1])
            z_values = openmc.stats.Discrete([self.RZ[1][i]], [1])
            angle = openmc.stats.Uniform(a=angles[0], b=angles[1])

            # create a ring source
            my_source.space = openmc.stats.CylindricalIndependent(
                r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0))

            my_source.angle = openmc.stats.Isotropic()
            my_source.energy = openmc.stats.Muir(
                e0=14080000.0, m_rat=5.0, kt=self.temperatures[i])

            # the strength of the source (its probability) is given by
            # self.strengths
            my_source.strength = self.strengths[i]

            # append to the list of sources
            sources.append(my_source)
        return sources


def strength(ion_density, ion_temperature):
    """Computes the neutron source density given ion density and ion
    temperature.

    Args:
        ion_density (float, ndarray): Ion density (m-3)
        ion_temperature (float, ndarray): Ion temperature (keV)

    Returns:
        float, ndarray: Neutron source density (neutron/s/m3)
    """
    return ion_density**2*DT_xs(ion_temperature)


def DT_xs(ion_temperature):
    """Sadler–Van Belle formula
    Ref : https://doi.org/10.1016/j.fusengdes.2012.02.025

    Args:
        ion_temperature (float, ndarray): ion temperature in keV

    Returns:
        float, ndarray: the DT cross section at the given temperature
    """
    c = [
        2.5663271e-18,
        19.983026,
        2.5077133e-2,
        2.5773408e-3,
        6.1880463e-5,
        6.6024089e-2,
        8.1215505e-3
        ]
    prod = ion_temperature*(c[2]+ion_temperature*(c[3]-c[4]*ion_temperature))
    prod *= 1/(1.0+ion_temperature*(c[5]+c[6]*ion_temperature))
    U = 1 - prod

    val = c[0]/(U**(5/6)*ion_temperature**(2/3))
    val *= np.exp(-c[1]*(U/ion_temperature)**(1/3))
    return val


if __name__ == "__main__":
    my_plasma = Plasma(
        elongation=1.557,
        ion_density_centre=1.09e20,
        ion_density_peaking_factor=1,
        ion_density_pedestal=1.09e20,
        ion_density_separatrix=3e19,
        ion_temperature_centre=45.9,
        ion_temperature_peaking_factor=8.06,
        ion_temperature_pedestal=6.09,
        ion_temperature_separatrix=0.1,
        major_radius=9.06,
        minor_radius=2.92258,
        pedestal_radius=0.8 * 2.92258,
        mode="H",
        shafranov_factor=0.44789,
        triangularity=0.270,
        ion_temperature_beta=6
        )
    my_plasma.sample_sources()
    fig, axs = plt.subplots(1, 2)
    plt.sca(axs[0])
    plt.scatter(my_plasma.RZ[0], my_plasma.RZ[1], c=my_plasma.strengths)
    plt.colorbar(label="Neutron source density (neutron/s/m3)")
    plt.sca(axs[1])
    plt.scatter(my_plasma.RZ[0], my_plasma.RZ[1], c=my_plasma.temperatures, cmap=cm.plasma)
    plt.colorbar(label="Ion temperature (keV)")

    for ax in axs:
        ax.set_aspect("equal")
    plt.show()
