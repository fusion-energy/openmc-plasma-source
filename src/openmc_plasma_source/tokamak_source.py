from typing import Tuple
import math
import numpy as np
import openmc
import openmc.checkvalue as cv
import NeSST as nst
from .fuel_types import get_neutron_energy_distribution, get_reactions_from_fuel
from NeSST.spectral_model import reac_DD, reac_TT, reac_DT


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
            centre (eV)
        ion_temperature_peaking_factor (float): Ion temperature peaking
            factor (referred in [1] as ion temperature exponent alpha_T)
        ion_temperature_beta (float): Ion temperature beta exponent
            (referred in [1] as ion temperature exponent beta_T)
        ion_temperature_pedestal (float): Ion temperature at pedestal (eV)
        ion_temperature_separatrix (float): Ion temperature at separatrix
            (eV)
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
    a = np.linspace(0, minor_radius)

    cylindrical_mesh = openmc.CylindricalMesh(
        r_grid=np.linspace(0, major_radius + minor_radius, 3),
        phi_grid=np.linspace(angles[0], angles[1], 2),
        z_grid=np.linspace(
            -1 * elongation * minor_radius, elongation * minor_radius, 4
        ),
    )
    r_vals = []
    z_vals = []
    a_vals = []
    for coords in cylindrical_mesh.centroids.reshape(
        np.prod(cylindrical_mesh.dimension), 3
    ):
        print(coords)
        radial = math.sqrt(coords[0] ** 2 + coords[1] ** 2)
        r_vals.append(radial)
        z_vals.append(coords[2])
        a_vals.append(abs(radial - major_radius))

    a = np.array(a_vals)
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

    fuel_densities = {}
    for key, value in fuel.items():
        fuel_densities[key] = densities * value

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

    reactions = get_reactions_from_fuel(fuel)

    neutron_source_density = {}
    total_source_density = 0
    for reaction in reactions:
        neutron_source_density[reaction] = tokamak_neutron_source_density(
            fuel_densities, temperatures, reaction
        )
        total_source_density += sum(neutron_source_density[reaction])

    all_sources = []
    for reaction in reactions:
        neutron_source_density[reaction] = (
            neutron_source_density[reaction] / total_source_density
        )

        sources = tokamak_make_openmc_sources(
            strengths=neutron_source_density,
            temperatures=temperatures,
            fuel=fuel,
        )
        all_sources = all_sources + sources

    mesh_source = openmc.MeshSource(
        mesh=cylindrical_mesh,
        sources=np.array(all_sources).reshape(cylindrical_mesh.dimension),
    )
    return mesh_source


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
    temperatures,
    fuel,
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
    for i, temperature in enumerate(temperatures):
        # extract the RZ values accordingly

        energy_distributions_and_dist_strengths = get_neutron_energy_distribution(
            ion_temperature=temperature,
            fuel=fuel,
        )

        # now we have potentially 3 distributions (DT, DD, TT)
        dists_for_mesh_voxel = []
        probs_for_mesh_voxel = []
        for reaction, (
            energy_distribution,
            dist_strength,
        ) in energy_distributions_and_dist_strengths.items():

            # the strength of the source (its probability) is given by the
            # strength of the energy distribution and the location distribution
            probs_for_mesh_voxel.append(
                dist_strength * strengths[reaction][i]
            )  # todo check that prob of 1 and set strength gives right answer in combined distribution

            dists_for_mesh_voxel.append(energy_distribution)

        source_for_mesh_voxel = openmc.data.combine_distributions(
            dists_for_mesh_voxel, probs_for_mesh_voxel
        )
        my_source = openmc.IndependentSource()

        my_source.angle = openmc.stats.Isotropic()

        my_source.energy = source_for_mesh_voxel
        sources.append(source_for_mesh_voxel)
    return sources


def tokamak_neutron_source_density(ion_density, ion_temperature, reaction):
    """Computes the neutron source density given ion density and ion
    temperature.

    Args:
        ion_density (float, ndarray): Ion density (m-3)
        ion_temperature (float, ndarray): Ion temperature (keV)

    Returns:
        float, ndarray: Neutron source density (neutron/s/m3)
    """

    ion_density = np.asarray(ion_density[reaction[0]]) * np.asarray(
        ion_density[reaction[1]]
    )
    ion_temperature = np.asarray(ion_temperature)

    if reaction == ["DD"]:
        return ion_density * reac_DD(ion_temperature)
    elif reaction == ["TT"]:
        return ion_density * reac_TT(ion_temperature)
    # ['DT', 'DD', 'TT']
    else:
        return ion_density * reac_DT(ion_temperature)  # could use _DT_xs instead


# TODO consider replace with NeSST or getting DD version as well
def _DT_xs(ion_temperature):
    """Sadler–Van Belle formula
    Ref : https://doi.org/10.1016/j.fusengdes.2012.02.025
    Args:
        ion_temperature (float, ndarray): ion temperature in eV
    Returns:
        float, ndarray: the DT cross section at the given temperature
    """
    ion_temperature_kev = np.asarray(ion_temperature / 1e3)
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
