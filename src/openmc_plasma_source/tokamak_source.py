from typing import Tuple, Dict, List, Union

import numpy as np
from numpy.typing import NDArray
import openmc
from openmc import IndependentSource
import openmc.checkvalue as cv
from NeSST.spectral_model import reac_DD, reac_DT, reac_TT

from .fuel_types import get_neutron_energy_distribution, get_reactions_from_fuel


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
    sample_size: int = 10000,
    fuel: Dict[str, float] = {"D": 0.5, "T": 0.5},
    sample_seed: int = 122807528840384100672342137672332424406,
) -> List[IndependentSource]:
    """
    Creates a list of openmc.IndependentSource objects representing a tokamak plasma.

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
        major_radius: Plasma major radius (cm)
        minor_radius: Plasma minor radius (cm)
        elongation: Plasma elongation
        triangularity: Plasma triangularity
        mode: Confinement mode ("L", "H", "A")
        ion_density_centre: Ion density at the plasma centre (m-3)
        ion_density_peaking_factor: Ion density peaking factor
            (referred in [1] as ion density exponent)
        ion_density_pedestal: Ion density at pedestal (m-3)
        ion_density_separatrix: Ion density at separatrix (m-3)
        ion_temperature_centre: Ion temperature at the plasma
            centre (eV)
        ion_temperature_peaking_factor: Ion temperature peaking
            factor (referred in [1] as ion temperature exponent alpha_T)
        ion_temperature_beta: Ion temperature beta exponent
            (referred in [1] as ion temperature exponent beta_T)
        ion_temperature_pedestal: Ion temperature at pedestal (eV)
        ion_temperature_separatrix: Ion temperature at separatrix
            (eV)
        pedestal_radius: Minor radius at pedestal (cm)
        shafranov_factor: Shafranov factor (referred in [1] as esh)
            also known as outward radial displacement of magnetic surfaces
            (cm)
        angles: the start and stop angles of the ring in
            radians
        sample_seed int: the seed passed to numpy.random when sampling source
            location. Numpy recommend a large int value. Defaults to
            122807528840384100672342137672332424406
        sample_size: number of neutron sources. Defaults
            to 1000.
        fuel: Isotopes as keys and atom fractions as values
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

    if not (
        isinstance(angles, tuple)
        and len(angles) == 2
        and all(
            isinstance(angle, (int, float)) and -2 * np.pi <= angle <= 2 * np.pi
            for angle in angles
        )
    ):
        raise ValueError("Angles must be a tuple of floats between zero and 2 * np.pi")

    # Create a list of sources
    """Samples sample_size neutrons and creates attributes .densities
    (ion density), .temperatures (ion temperature), .strengths
    (neutron source density) and .RZ (coordinates)
    """
    # create a sample of (a, alpha) coordinates
    rng = np.random.default_rng(sample_seed)
    a = rng.random(sample_size) * minor_radius
    alpha = rng.random(sample_size) * 2 * np.pi

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
        ion_temperature_pedestal=ion_temperature_pedestal / 1e3,
        ion_temperature_centre=ion_temperature_centre / 1e3,
        ion_temperature_beta=ion_temperature_beta,
        ion_temperature_peaking_factor=ion_temperature_peaking_factor,
        ion_temperature_separatrix=ion_temperature_separatrix / 1e3,
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

    fuel_densities = {}
    for key, value in fuel.items():
        fuel_densities[key] = densities * value
    reactions = get_reactions_from_fuel(fuel)

    neutron_source_density = {}  # type: Dict[str, NDArray]
    total_source_density = 0.0
    for reaction in reactions:

        if reaction == "DD":
            fuel_density = fuel_densities["D"] * 0.5
        elif reaction == "TT":
            fuel_density = fuel_densities["T"] * 0.5
        elif reaction == "DT":
            fuel_density = fuel_densities["T"] * fuel_densities["D"]

        neutron_source_density[reaction] = tokamak_neutron_source_density(
            fuel_density, temperatures, reaction
        )
        if reaction == "TT":
            # TT reaction emits 2 neutrons
            neutron_source_density[reaction] = neutron_source_density[reaction] * 2

        total_source_density += sum(neutron_source_density[reaction])

    all_sources = []  # type: List[IndependentSource]
    for reaction in reactions:
        strengths = neutron_source_density[reaction] / total_source_density

        sources = tokamak_make_openmc_sources(
            strengths=strengths,
            angles=angles,
            temperatures=temperatures,
            fuel=fuel,
            RZ=RZ,
        )
        all_sources = all_sources + sources
    return all_sources


def tokamak_ion_density(
    mode: str,
    ion_density_centre: float,
    ion_density_peaking_factor: float,
    ion_density_pedestal: float,
    major_radius: float,
    pedestal_radius: float,
    ion_density_separatrix: float,
    r: Union[float, NDArray],
) -> NDArray:
    """
    Computes the ion density at a given position. The ion density is
    only dependent on the minor radius.

    Args:
        mode: Confinement mode ("L", "H", "A")
        ion_density_centre: Ion density at the plasma centre (m-3)
        ion_density_peaking_factor: Ion density peaking factor
            (referred in [1] as ion density exponent)
        ion_density_pedestal: Ion density at pedestal (m-3)
        major_radius: Plasma major radius (cm)
        pedestal_radius: Minor radius at pedestal (cm)
        ion_density_separatrix: Ion density at separatrix (m-3)
        r: Minor radius (cm)

    Returns:
        ion density in m-3
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
    r: Union[float, NDArray],
    mode: str,
    pedestal_radius: float,
    ion_temperature_pedestal: float,
    ion_temperature_centre: float,
    ion_temperature_beta: float,
    ion_temperature_peaking_factor: float,
    ion_temperature_separatrix: float,
    major_radius: float,
) -> NDArray:
    """
    Computes the ion temperature at a given position. The ion
    temperature is only dependent on the minor radius.

    Args:
        r: Minor radius (cm)
        mode: Confinement mode ("L", "H", "A")
        pedestal_radius: Minor radius at pedestal (cm)
        ion_temperature_pedestal: Ion temperature at pedestal (eV)
        ion_temperature_centre: Ion temperature at the plasma
            centre (eV)
        ion_temperature_beta: Ion temperature beta exponent
        ion_temperature_peaking_factor: Ion temperature peaking
            factor
        ion_temperature_separatrix: Ion temperature at separatrix (eV)
        major_radius: Plasma major radius (cm)


    Returns:
        ion temperature (eV)
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
    return temperature * 1e3


def tokamak_convert_a_alpha_to_R_Z(
    a: Union[float, NDArray],
    alpha: Union[float, NDArray],
    shafranov_factor: float,
    minor_radius: float,
    major_radius: float,
    triangularity: float,
    elongation: float,
) -> Tuple[NDArray, NDArray]:
    """
    Converts (r, alpha) cylindrical coordinates to (R, Z) cartesian
    coordinates.

    Args:
        a: Minor radius (cm)
        alpha: Poloidal angle (radians)
        shafranov_factor: Shafranov factor
        minor_radius: Plasma minor radius (cm)
        major_radius: Plasma major radius (cm)
        triangularity: Plasma triangularity
        elongation: Plasma elongation

    Returns:
        (R, Z) coordinates
    """
    a = np.asarray(a)
    alpha = np.asarray(alpha)
    if np.any(a < 0):
        raise ValueError("Radius 'a' must not be negative")

    shafranov_shift = shafranov_factor * (1.0 - (a / minor_radius) ** 2)
    R = (
        major_radius
        + a * np.cos(alpha + (triangularity * np.sin(alpha)))
        + shafranov_shift
    )
    Z = elongation * a * np.sin(alpha)
    return (R, Z)


def tokamak_make_openmc_sources(
    strengths: List[float] | NDArray,
    angles: Tuple[float, float],
    temperatures: NDArray,
    fuel: Dict[str, float],
    RZ: Tuple[NDArray, NDArray],
) -> List[IndependentSource]:
    """
    Creates a list of OpenMC IndependentSource() objects. The created sources are
    ring sources based on the .RZ coordinates between two angles. The
    energy of the sources are Muir energy spectra with ion temperatures
    based on .temperatures. The strength of the sources (their probability)
    is based on .strengths.

    Args:
        strengths: The strength of the sources
        angles: The start and stop angles of the ring in radians
        temperatures: The ion temperatures
        fuel: Isotopes as keys and atom fractions as values
        RZ: The (R, Z) coordinates of the sources

    Returns:
        list of openmc.IndependentSource instances
    """

    sources = []
    # create a ring source for each sample in the plasma source
    R_vals = RZ[0]
    Z_vals = RZ[1]
    assert len(Z_vals) == len(R_vals) == len(temperatures) == len(strengths)
    for R_val, Z_val, temperature, strength in zip(
        R_vals, Z_vals, temperatures, strengths
    ):

        if strength > 0.0:
            radius = openmc.stats.Discrete([R_val], [1])
            z_values = openmc.stats.Discrete([Z_val], [1])
            angle = openmc.stats.Uniform(a=angles[0], b=angles[1])

            my_source = IndependentSource()
            my_source.energy = get_neutron_energy_distribution(
                ion_temperature=temperature,
                fuel=fuel,
            )

            # create a ring source
            my_source.space = openmc.stats.CylindricalIndependent(
                r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0)
            )
            my_source.angle = openmc.stats.Isotropic()

            my_source.strength = strength

            # append to the list of sources
            sources.append(my_source)
    return sources


def tokamak_neutron_source_density(
    ion_density: Union[float, NDArray],
    ion_temperature: Union[float, NDArray],
    reaction: str,
) -> NDArray:
    """
    Computes the neutron source density given ion density and ion
    temperature.

    Args:
        ion_density: Ion density (m-3)
        ion_temperature: Ion temperature (eV)
        reaction: The fusion reactions to consider e.g. 'DD'

    Returns:
        Neutron source density (neutron/s/m3)
    """

    ion_density = np.asarray(ion_density)
    ion_temperature = np.asarray(ion_temperature)

    if reaction == "DD":
        return ion_density * reac_DD(ion_temperature)
    elif reaction == "TT":
        return ion_density * reac_TT(ion_temperature)
    elif reaction == "DT":
        return ion_density * reac_DT(ion_temperature)  # could use _DT_xs instead
    else:
        raise ValueError(
            f'Reaction {reaction} not in available options ["DD", "DT", "TT"]'
        )
