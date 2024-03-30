from typing import Tuple

import numpy as np
import openmc
import openmc.checkvalue as cv

from .fuel_types import get_neutron_energy_distribution


def tokamak_source(
    major_radius: float,
    minor_radius: float,
    elongation: float,
    triangularity: float,
    mode: str,
    ion_density_centre: float,
    ion_density_peaking_factor: float,
    ion_density_pedestal: float,
    ion_density_separatrix: float,
    ion_temperature_centre: float,
    ion_temperature_peaking_factor: float,
    ion_temperature_beta: float,
    ion_temperature_pedestal: float,
    ion_temperature_separatrix: float,
    pedestal_radius: float,
    shafranov_factor: float,
    angles: Tuple[float, float] = (0, 2 * np.pi),
    sample_size: int = 1000,
    fuel: dict = {"D": 0.5, "T": 0.5},
) -> list[openmc.IndependentSource]:
    """Creates a list of openmc.IndependentSource objects representing a tokamak plasma.

    Resulting sources will have an energy distribution according to the fuel
    composition.This function greatly relies on models described in [1]

    [1] : Fausser et al, 'Tokamak D-T neutron source models for different
    plasma physics confinement modes', Fus. Eng. and Design,
    https://doi.org/10.1016/j.fusengdes.2012.02.025

    Usage:
        my_source = tokamak_source(**plasma_prms)
        my_settings = openmc.Settings()
        my_settings.source = my_source

    Args:
        major_radius (float): Plasma major radius (cm)
        minor_radius (float): Plasma minor radius (cm)
        elongation (float): Plasma elongation
        triangularity (float): Plasma triangularity
        mode (str): Confinement mode ("L", "H", "A")
        ion_density_centre (float): Ion density at the plasma centre (m-3)
        ion_density_peaking_factor (float): Ion density peaking factor
            (referred in [1] as ion density exponent)
        ion_density_pedestal (float): Ion density at pedestal (m-3)
        ion_density_separatrix (float): Ion density at separatrix (m-3)
        ion_temperature_centre (float): Ion temperature at the plasma
            centre (keV)
        ion_temperature_peaking_factor (float): Ion temperature peaking
            factor (referred in [1] as ion temperature exponent alpha_T)
        ion_temperature_beta (float): Ion temperature beta exponent
            (referred in [1] as ion temperature exponent beta_T)
        ion_temperature_pedestal (float): Ion temperature at pedestal (keV)
        ion_temperature_separatrix (float): Ion temperature at separatrix
            (keV)
        pedestal_radius (float): Minor radius at pedestal (cm)
        shafranov_factor (float): Shafranov factor (referred in [1] as esh)
            also known as outward radial displacement of magnetic surfaces
            (cm)
        angles (iterable of floats): the start and stop angles of the ring in
            radians
        sample_size (int, optional): number of neutron sources. Defaults
            to 1000.
        fuel (dict): Isotopes as keys and atom fractions as values
    """

    # Perform sanity checks for inputs not caught by properties
    cv.check_type("major_radius", major_radius, (int, float))
    cv.check_type("minor_radius", minor_radius, (int, float))
    cv.check_type("elongation", elongation, (int, float))
    cv.check_type("triangularity", triangularity, (int, float))
    cv.check_type("ion_density_centre", ion_density_centre, (int, float))
    cv.check_type(
        "ion_density_peaking_factor", ion_density_peaking_factor, (int, float)
    )
    cv.check_type("ion_density_pedestal", ion_density_pedestal, (int, float))
    cv.check_type("ion_density_separatrix", ion_density_separatrix, (int, float))
    cv.check_less_than("minor_radius", minor_radius, major_radius)
    cv.check_less_than("pedestal_radius", pedestal_radius, minor_radius)
    cv.check_less_than("shafranov_factor", abs(shafranov_factor), 0.5 * minor_radius)
    cv.check_greater_than("major_radius", major_radius, 0)
    cv.check_greater_than("minor_radius", minor_radius, 0)
    cv.check_greater_than("elongation", elongation, 0)
    cv.check_less_than("triangularity", triangularity, 1.0, True)
    cv.check_greater_than("triangularity", triangularity, -1.0, True)
    cv.check_value("mode", mode, ["H", "L", "A"])
    cv.check_greater_than("ion_density_centre", ion_density_centre, 0)
    cv.check_greater_than("ion_density_pedestal", ion_density_pedestal, 0)
    cv.check_greater_than("ion_density_separatrix", ion_density_separatrix, 0)

    if (
        isinstance(angles, tuple)
        and len(angles) == 2
        and all(
            isinstance(angle, (int, float)) and -2 * np.pi <= angle <= 2 * np.pi
            for angle in angles
        )
    ):
        pass
    else:
        raise ValueError("Angles must be a tuple of floats between zero and 2 * np.pi")

    # Create a list of sources
    """Samples sample_size neutrons and creates attributes .densities
    (ion density), .temperatures (ion temperature), .strengths
    (neutron source density) and .RZ (coordinates)
    """
    # create a sample of (a, alpha) coordinates
    a = np.random.random(sample_size) * minor_radius
    alpha = np.random.random(sample_size) * 2 * np.pi

    # compute densities, temperatures
    densities = tokamak_ion_density(
        mode=mode,
        ion_density_centre=ion_density_centre,
        ion_density_peaking_factor=ion_density_peaking_factor,
        ion_density_pedestal=ion_density_pedestal,
        major_radius=major_radius,
        pedestal_radius=pedestal_radius,
        ion_density_separatrix=ion_density_separatrix,
        r=a,
    )

    # compute temperatures
    temperatures = tokamak_ion_temperature(
        r=a,
        mode=mode,
        pedestal_radius=pedestal_radius,
        ion_temperature_pedestal=ion_temperature_pedestal,
        ion_temperature_centre=ion_temperature_centre,
        ion_temperature_beta=ion_temperature_beta,
        ion_temperature_peaking_factor=ion_temperature_peaking_factor,
        ion_temperature_separatrix=ion_temperature_separatrix,
        major_radius=major_radius,
    )

    # convert coordinates
    RZ = tokamak_convert_a_alpha_to_R_Z(
        a=a,
        alpha=alpha,
        shafranov_factor=shafranov_factor,
        minor_radius=minor_radius,
        major_radius=major_radius,
        triangularity=triangularity,
        elongation=elongation,
    )

    #TODO loop through the fuel reactions

    # compute neutron source densities
    # loop through each reaction in the fuel
    # could introduce a species density

    for 
        neutron_source_density = tokamak_neutron_source_density(densities, temperatures, reaction=, fuel)

        strengths = neutron_source_density / sum(neutron_source_density)


        sources = tokamak_make_openmc_sources(
            strengths=strengths,
            angles=angles,
            temperatures=temperatures,
            fuel=fuel,
            RZ=RZ,
        )
        return sources


def tokamak_ion_density(
    mode,
    ion_density_centre,
    ion_density_peaking_factor,
    ion_density_pedestal,
    major_radius,
    pedestal_radius,
    ion_density_separatrix,
    r,
):
    """Computes the ion density at a given position. The ion density is
    only dependent on the minor radius.

    Args:
        r (float, ndarray): the minor radius (cm)

    Returns:
        float, ndarray: ion density in m-3
    """

    r = np.asarray(r)
    if np.any(r < 0):
        raise ValueError("Minor radius must not be negative")

    if mode == "L":
        density = (
            ion_density_centre
            * (1 - (r / major_radius) ** 2) ** ion_density_peaking_factor
        )
    elif mode in ["H", "A"]:
        density = np.where(
            r < pedestal_radius,
            (
                (ion_density_centre - ion_density_pedestal)
                * (1 - (r / pedestal_radius) ** 2) ** ion_density_peaking_factor
                + ion_density_pedestal
            ),
            (
                (ion_density_pedestal - ion_density_separatrix)
                * (major_radius - r)
                / (major_radius - pedestal_radius)
                + ion_density_separatrix
            ),
        )
    return density


def tokamak_ion_temperature(
    r,
    mode,
    pedestal_radius,
    ion_temperature_pedestal,
    ion_temperature_centre,
    ion_temperature_beta,
    ion_temperature_peaking_factor,
    ion_temperature_separatrix,
    major_radius,
):
    """Computes the ion temperature at a given position. The ion
    temperature is only dependent on the minor radius.

    Args:
        r (float, ndarray): minor radius (cm)

    Returns:
        float, ndarray: ion temperature (keV)
    """

    r = np.asarray(r)
    if np.any(r < 0):
        raise ValueError("Minor radius must not be negative")

    if mode == "L":
        temperature = (
            ion_temperature_centre
            * (1 - (r / major_radius) ** 2) ** ion_temperature_peaking_factor
        )
    elif mode in ["H", "A"]:
        temperature = np.where(
            r < pedestal_radius,
            (
                ion_temperature_pedestal
                + (ion_temperature_centre - ion_temperature_pedestal)
                * (1 - (np.abs(r / pedestal_radius)) ** ion_temperature_beta)
                ** ion_temperature_peaking_factor
            ),
            (
                ion_temperature_separatrix
                + (ion_temperature_pedestal - ion_temperature_separatrix)
                * (major_radius - r)
                / (major_radius - pedestal_radius)
            ),
        )
    return temperature


def tokamak_convert_a_alpha_to_R_Z(
    a,
    alpha,
    shafranov_factor,
    minor_radius,
    major_radius,
    triangularity,
    elongation,
):
    """Converts (r, alpha) cylindrical coordinates to (R, Z) cartesian
    coordinates.

    Args:
        a (float, ndarray): minor radius (cm)
        alpha (float, ndarray): angle (rad)
        shafranov_factor:
        minor_radius:
        major_radius:

    Returns:
        ((float, ndarray), (float, ndarray)): (R, Z) coordinates
    """
    a = np.asarray(a)
    alpha = np.asarray(alpha)
    if np.any(a < 0):
        raise ValueError("Radius 'a'  must not be negative")

    shafranov_shift = shafranov_factor * (1.0 - (a / minor_radius) ** 2)
    R = (
        major_radius
        + a * np.cos(alpha + (triangularity * np.sin(alpha)))
        + shafranov_shift
    )
    Z = elongation * a * np.sin(alpha)
    return (R, Z)


def tokamak_make_openmc_sources(
    strengths,
    angles,
    temperatures,
    fuel,
    RZ,
):
    """Creates a list of OpenMC Sources() objects. The created sources are
    ring sources based on the .RZ coordinates between two angles. The
    energy of the sources are Muir energy spectra with ion temperatures
    based on .temperatures. The strength of the sources (their probability)
    is based on .strengths.

    Args:
        strengths
        angles ((float, float), optional): rotation of the ring source.
            Defaults to (0, 2*np.pi).
        temperatures
        fuel
        RZ

    Returns:
        list: list of openmc.IndependentSource()
    """

    sources = []
    # create a ring source for each sample in the plasma source
    for i in range(len(strengths)):
        # extract the RZ values accordingly
        radius = openmc.stats.Discrete([RZ[0][i]], [1])
        z_values = openmc.stats.Discrete([RZ[1][i]], [1])
        angle = openmc.stats.Uniform(a=angles[0], b=angles[1])
        strength = strengths[i]

        energy_distributions, dist_strengths = get_neutron_energy_distribution(
            ion_temperature=temperatures[i],
            fuel=fuel,
        )

        # now we have potentially 3 distributions (DT, DD, TT)
        for energy_distribution, dist_strength in zip(
            energy_distributions, dist_strengths
        ):
            my_source = openmc.IndependentSource()

            # create a ring source
            my_source.space = openmc.stats.CylindricalIndependent(
                r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0)
            )
            my_source.angle = openmc.stats.Isotropic()

            my_source.energy = energy_distribution

            # the strength of the source (its probability) is given by the
            # strength of the energy distribution and the location distribution
            my_source.strength = dist_strength * strength

            # append to the list of sources
            sources.append(my_source)
    return sources


def tokamak_neutron_source_density(ion_density, ion_temperature, reaction='DT'):
    """Computes the neutron source density given ion density and ion
    temperature.

    Args:
        ion_density (float, ndarray): Ion density (m-3)
        ion_temperature (float, ndarray): Ion temperature (keV)

    Returns:
        float, ndarray: Neutron source density (neutron/s/m3)
    """

    ion_density = np.asarray(ion_density)
    ion_temperature = np.asarray(ion_temperature)
    return ion_density**2 * _DT_xs(ion_temperature)
    #TODO account for reaction
    #TODO get a fuel fraction from fuel va
    if reaction == 'DT':
        ion_density**2 * nst.reac_DT(ion_temperature) # could use _DT_xs instead
    if reaction == 'DD':
        ion_density**2 * nst.reac_DD(ion_temperature)
    if reaction == 'TT':
        ion_density**2 * nst.reac_TT(ion_temperature)


#TODO consider replace with NeSST or getting DD version as well
def _DT_xs(ion_temperature):
    """Sadler–Van Belle formula
    Ref : https://doi.org/10.1016/j.fusengdes.2012.02.025
    Args:
        ion_temperature (float, ndarray): ion temperature in eV
    Returns:
        float, ndarray: the DT cross section at the given temperature
    """
    ion_temperature_kev = np.asarray(ion_temperature/1e3)
    c = [
        2.5663271e-18,
        19.983026,
        2.5077133e-2,
        2.5773408e-3,
        6.1880463e-5,
        6.6024089e-2,
        8.1215505e-3,
    ]
    U = 1 - ion_temperature_kev * (
        c[2] + ion_temperature_kev * (c[3] - c[4] * ion_temperature_kev)
    ) / (1.0 + ion_temperature_kev * (c[5] + c[6] * ion_temperature_kev))
    val = (
        c[0]
        * np.exp(-c[1] * (U / ion_temperature_kev) ** (1 / 3))
        / (U ** (5 / 6) * ion_temperature_kev ** (2 / 3))
    )
    return val
