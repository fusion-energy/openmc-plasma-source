from typing import Tuple, Dict, List, Union

import numpy as np
from numpy.typing import NDArray
import openmc
from openmc import IndependentSource
import openmc.checkvalue as cv
from NeSST.spectral_model import reac_DD, reac_DT, reac_TT

from .fuel_types import get_neutron_energy_distribution, get_reactions_from_fuel


def _toroidal_phi_grid(
    start_angle: float, rotation_angle: float, n_phi: int
) -> Tuple[NDArray, NDArray]:
    """Build a CylindricalMesh phi_grid and per-bin source strength fractions.

    The toroidal sector spans an angular width of abs(rotation_angle), divided
    into n_phi bins. A CylindricalMesh requires phi edges that are increasing
    and within [0, 2*pi], so when the sector crosses the 0 / 2*pi seam the grid
    covers the full circle with a zero-strength "dead" bin filling the
    unselected portion; otherwise it is a partial mesh covering only the sector.

    Args:
        start_angle: Toroidal angle at which the sector starts (radians).
        rotation_angle: Toroidal extent of the sector (radians). May be negative.
        n_phi: Number of bins requested across the sector.

    Returns:
        (phi_grid, phi_fraction) where phi_grid is the array of bin edges and
        phi_fraction is the fraction of the total source strength assigned to
        each bin (summing to 1 over the active bins, 0 in any dead bin).
    """
    two_pi = 2 * np.pi
    width = abs(rotation_angle)

    # A sector spanning the full circle is the whole torus regardless of start.
    if np.isclose(width, two_pi):
        phi_grid = np.linspace(0.0, two_pi, n_phi + 1)
        bin_widths = np.diff(phi_grid)
        return phi_grid, bin_widths / bin_widths.sum()

    # Normalize the sector start into [0, 2*pi); the unwrapped end may exceed it.
    lo = min(start_angle, start_angle + rotation_angle) % two_pi
    hi = lo + width
    raw = np.linspace(lo, hi, n_phi + 1)

    if hi <= two_pi + 1e-12:
        # Sector does not cross the seam: a partial mesh covering [lo, hi].
        phi_grid = np.clip(raw, 0.0, two_pi)
        bin_widths = np.diff(phi_grid)
        return phi_grid, bin_widths / width

    # Sector crosses the seam: cover the full circle. The selected angles are
    # [0, hi - 2*pi] (wrapped end) and [lo, 2*pi] (start); the gap between is a
    # single dead bin carrying zero strength.
    seam_hi = hi - two_pi
    lower = raw[raw < two_pi]  # edges from lo up towards 2*pi
    upper = raw[raw > two_pi] - two_pi  # wrapped edges from 0 up to seam_hi
    edges = np.concatenate(([0.0], upper, [seam_hi, lo], lower, [two_pi]))
    phi_grid = np.unique(edges)  # sorted and deduplicated

    bin_widths = np.diff(phi_grid)
    mids = 0.5 * (phi_grid[:-1] + phi_grid[1:])
    active = (mids <= seam_hi + 1e-12) | (mids >= lo - 1e-12)
    phi_fraction = np.where(active, bin_widths / width, 0.0)
    return phi_grid, phi_fraction


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
    start_angle: float = 0.0,
    rotation_angle: float = 2 * np.pi,
    mesh_resolution: Tuple[int, int, int] = (100, 1, 100),
    grid_density: int = 500,
    fuel: Dict[str, float] = {"D": 0.5, "T": 0.5},
) -> openmc.MeshSource:
    """
    Creates an openmc.MeshSource representing a tokamak plasma.

    Uses a CylindricalMesh to discretize the plasma cross-section. A dense
    grid in plasma (a, alpha) coordinates is forward-mapped to (R, Z) and
    binned into the mesh. The energy distribution at each mesh voxel is
    determined by the local ion temperature and fuel composition. This
    function greatly relies on models described in [1].

    [1] : Fausser et al, 'Tokamak D-T neutron source models for different
    plasma physics confinement modes', Fus. Eng. and Design,
    https://doi.org/10.1016/j.fusengdes.2012.02.025

    Usage:
        my_source = tokamak_source(**plasma_prms)
        my_settings = openmc.Settings()
        my_settings.source = [my_source]

    Args:
        major_radius: Plasma major radius (cm)
        minor_radius: Plasma minor radius (cm)
        elongation: Plasma elongation (dimensionless)
        triangularity: Plasma triangularity (dimensionless)
        mode: Confinement mode ("L", "H", "A")
        ion_density_centre: Ion density at the plasma centre (m-3)
        ion_density_peaking_factor: Ion density peaking factor
            (dimensionless, referred in [1] as ion density exponent)
        ion_density_pedestal: Ion density at pedestal (m-3)
        ion_density_separatrix: Ion density at separatrix (m-3)
        ion_temperature_centre: Ion temperature at the plasma
            centre (eV)
        ion_temperature_peaking_factor: Ion temperature peaking
            factor (dimensionless, referred in [1] as ion temperature
            exponent alpha_T)
        ion_temperature_beta: Ion temperature beta exponent
            (dimensionless, referred in [1] as ion temperature
            exponent beta_T)
        ion_temperature_pedestal: Ion temperature at pedestal (eV)
        ion_temperature_separatrix: Ion temperature at separatrix
            (eV)
        pedestal_radius: Minor radius at pedestal (cm)
        shafranov_factor: Shafranov factor (cm, referred in [1] as
            esh) also known as outward radial displacement of magnetic
            surfaces
        start_angle: The toroidal angle at which the plasma starts, in
            radians. Must be between -2*pi and 2*pi.
        rotation_angle: The toroidal extent of the plasma, in radians. Must
            be a non-zero value between -2*pi and 2*pi. A negative value
            extends the plasma in the opposite direction from start_angle. A
            sector that crosses the 0 / 2*pi seam is supported via a full-circle
            mesh with a zero-strength bin filling the unselected portion.
        mesh_resolution: Number of mesh bins in (r, phi, z).
            Defaults to (100, 1, 100).
        grid_density: Number of points per dimension in the internal
            (a, alpha) grid used for forward mapping. Defaults to 500.
        fuel: Isotopes as keys and atom fractions as values

    Returns:
        openmc.MeshSource backed by a CylindricalMesh
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
        isinstance(start_angle, (int, float))
        and not isinstance(start_angle, bool)
        and -2 * np.pi <= start_angle <= 2 * np.pi
    ):
        raise ValueError("start_angle must be a float between -2*pi and 2*pi")

    if not (
        isinstance(rotation_angle, (int, float))
        and not isinstance(rotation_angle, bool)
        and -2 * np.pi <= rotation_angle <= 2 * np.pi
        and rotation_angle != 0
    ):
        raise ValueError(
            "rotation_angle must be a non-zero float between -2*pi and 2*pi"
        )

    n_r, n_phi, n_z = mesh_resolution

    # Compute mesh bounds from geometry
    R_min = major_radius - minor_radius - abs(shafranov_factor)
    R_max = major_radius + minor_radius + abs(shafranov_factor)
    Z_max = elongation * minor_radius

    # Build the toroidal grid. phi_fraction holds the share of source strength
    # for each phi bin; bins outside the sector (the "dead" bin of a sector
    # that crosses the 0 / 2*pi seam) get zero.
    phi_grid, phi_fraction = _toroidal_phi_grid(start_angle, rotation_angle, n_phi)
    n_phi = len(phi_grid) - 1

    # Create cylindrical mesh
    mesh = openmc.CylindricalMesh(
        r_grid=np.linspace(R_min, R_max, n_r + 1),
        phi_grid=phi_grid,
        z_grid=np.linspace(-Z_max, Z_max, n_z + 1),
    )

    # Dense (a, alpha) grid for forward mapping
    da = minor_radius / grid_density
    dalpha = 2 * np.pi / grid_density
    a_centers = np.linspace(0.5 * da, minor_radius - 0.5 * da, grid_density)
    alpha_centers = np.linspace(0.5 * dalpha, 2 * np.pi - 0.5 * dalpha, grid_density)

    a_grid, alpha_grid = np.meshgrid(a_centers, alpha_centers, indexing="ij")
    a_flat = a_grid.ravel()
    alpha_flat = alpha_grid.ravel()

    # Compute densities and temperatures (depend only on minor radius a)
    densities = tokamak_ion_density(
        mode=mode,
        ion_density_centre=ion_density_centre,
        ion_density_peaking_factor=ion_density_peaking_factor,
        ion_density_pedestal=ion_density_pedestal,
        minor_radius=minor_radius,
        pedestal_radius=pedestal_radius,
        ion_density_separatrix=ion_density_separatrix,
        r=a_flat,
    )

    temperatures = tokamak_ion_temperature(
        r=a_flat,
        mode=mode,
        pedestal_radius=pedestal_radius,
        ion_temperature_pedestal=ion_temperature_pedestal / 1e3,
        ion_temperature_centre=ion_temperature_centre / 1e3,
        ion_temperature_beta=ion_temperature_beta,
        ion_temperature_peaking_factor=ion_temperature_peaking_factor,
        ion_temperature_separatrix=ion_temperature_separatrix / 1e3,
        minor_radius=minor_radius,
    )

    # Forward-map to (R, Z)
    R_flat, Z_flat = tokamak_convert_a_alpha_to_R_Z(
        a=a_flat,
        alpha=alpha_flat,
        shafranov_factor=shafranov_factor,
        minor_radius=minor_radius,
        major_radius=major_radius,
        triangularity=triangularity,
        elongation=elongation,
    )

    # Weight each grid point by the plasma volume element it represents. The
    # (a, alpha) grid is uniform, so to recover the true spatial neutron
    # emission each point must be weighted by the local volume per unit
    # (a, alpha): dV is proportional to R * |d(R,Z)/d(a,alpha)|, i.e. the
    # toroidal factor R times the poloidal cross-sectional Jacobian |J|.
    # Without the R factor the core (small a, small R) is over-represented.
    # R = R0 + a*cos(alpha + delta*sin(alpha)) + esh*(1 - (a/a_minor)^2)
    # Z = kappa * a * sin(alpha)
    theta = alpha_flat + triangularity * np.sin(alpha_flat)
    dtheta_dalpha = 1 + triangularity * np.cos(alpha_flat)
    dR_da = np.cos(theta) - 2 * shafranov_factor * a_flat / minor_radius**2
    dR_dalpha = -a_flat * np.sin(theta) * dtheta_dalpha
    dZ_da = elongation * np.sin(alpha_flat)
    dZ_dalpha = elongation * a_flat * np.cos(alpha_flat)
    jacobian = np.abs(dR_da * dZ_dalpha - dR_dalpha * dZ_da)
    volume_weight = R_flat * jacobian

    # Compute total neutron source density across all reactions
    fuel_densities = {key: densities * value for key, value in fuel.items()}
    reactions = get_reactions_from_fuel(fuel)

    total_source_density = np.zeros_like(a_flat)
    for reaction in reactions:
        if reaction == "DD":
            fuel_density = 0.5 * fuel_densities["D"] ** 2
        elif reaction == "TT":
            fuel_density = 0.5 * fuel_densities["T"] ** 2
        elif reaction == "DT":
            fuel_density = fuel_densities["T"] * fuel_densities["D"]

        nsd = tokamak_neutron_source_density(fuel_density, temperatures, reaction)
        if reaction == "TT":
            nsd = nsd * 2
        total_source_density += nsd

    # Bin source density and temperature into mesh cells, weighting each grid
    # point by the plasma volume element (toroidal R * poloidal Jacobian).
    weights = total_source_density * volume_weight * da * dalpha
    temp_weights = temperatures * weights

    r_edges = np.asarray(mesh.r_grid)
    z_edges = np.asarray(mesh.z_grid)

    binned_strength, _, _ = np.histogram2d(
        R_flat, Z_flat, bins=[r_edges, z_edges], weights=weights
    )
    binned_temp_weight, _, _ = np.histogram2d(
        R_flat, Z_flat, bins=[r_edges, z_edges], weights=temp_weights
    )

    # Source-density-weighted mean temperature per cell
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_temperature = np.where(
            binned_strength > 0,
            binned_temp_weight / binned_strength,
            0.0,
        )

    # Normalize strengths so they sum to 1
    total = binned_strength.sum()
    if total <= 0.0:
        raise ValueError(
            "Total neutron source density is zero. This may be caused by "
            "ion temperatures or densities that are too low to produce fusion reactions."
        )
    binned_strength /= total

    # Create one IndependentSource per mesh voxel
    sources = np.empty((n_r, n_phi, n_z), dtype=object)
    for i in range(n_r):
        for j in range(n_phi):
            for k in range(n_z):
                src = IndependentSource()
                strength = float(binned_strength[i, k]) * phi_fraction[j]
                temp = float(mean_temperature[i, k])

                if strength > 0 and temp > 0:
                    src.energy = get_neutron_energy_distribution(
                        ion_temperature=temp,
                        fuel=fuel,
                    )
                    src.angle = openmc.stats.Isotropic()

                src.strength = strength
                sources[i, j, k] = src

    return openmc.MeshSource(mesh, sources)


def tokamak_ion_density(
    mode: str,
    ion_density_centre: float,
    ion_density_peaking_factor: float,
    ion_density_pedestal: float,
    minor_radius: float,
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
            (dimensionless, referred in [1] as ion density exponent)
        ion_density_pedestal: Ion density at pedestal (m-3)
        minor_radius: Plasma minor radius (cm)
        pedestal_radius: Minor radius at pedestal (cm)
        ion_density_separatrix: Ion density at separatrix (m-3)
        r: Local minor radius (cm)

    Returns:
        ion density (m-3)
    """

    r = np.asarray(r)
    if np.any(r < 0):
        raise ValueError("Minor radius must not be negative")

    if mode == "L":
        density = (
            ion_density_centre
            * (1 - (r / minor_radius) ** 2) ** ion_density_peaking_factor
        )
    elif mode in ["H", "A"]:
        r_core = np.minimum(r, pedestal_radius)
        density = np.where(
            r < pedestal_radius,
            (
                (ion_density_centre - ion_density_pedestal)
                * (1 - (r_core / pedestal_radius) ** 2) ** ion_density_peaking_factor
                + ion_density_pedestal
            ),
            (
                (ion_density_pedestal - ion_density_separatrix)
                * (minor_radius - r)
                / (minor_radius - pedestal_radius)
                + ion_density_separatrix
            ),
        )
    else:
        raise ValueError(f'Mode {mode} not in available options ["L", "H", "A"]')
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
    minor_radius: float,
) -> NDArray:
    """
    Computes the ion temperature at a given position. The ion
    temperature is only dependent on the minor radius.

    Args:
        r: Local minor radius (cm)
        mode: Confinement mode ("L", "H", "A")
        pedestal_radius: Minor radius at pedestal (cm)
        ion_temperature_pedestal: Ion temperature at pedestal (keV)
        ion_temperature_centre: Ion temperature at the plasma
            centre (keV)
        ion_temperature_beta: Ion temperature beta exponent
            (dimensionless)
        ion_temperature_peaking_factor: Ion temperature peaking
            factor (dimensionless)
        ion_temperature_separatrix: Ion temperature at separatrix
            (keV)
        minor_radius: Plasma minor radius (cm)

    Returns:
        ion temperature (eV)
    """

    r = np.asarray(r)
    if np.any(r < 0):
        raise ValueError("Minor radius must not be negative")

    if mode == "L":
        temperature = (
            ion_temperature_centre
            * (1 - (r / minor_radius) ** 2) ** ion_temperature_peaking_factor
        )
    elif mode in ["H", "A"]:
        # Clip r to pedestal_radius in the core branch to avoid raising
        # a negative base to a non-integer power (which produces NaN).
        # np.where evaluates both branches for all elements, so without
        # clipping the r > pedestal_radius elements trigger a warning
        # even though their values are discarded.
        r_core = np.minimum(r, pedestal_radius)
        temperature = np.where(
            r < pedestal_radius,
            (
                ion_temperature_pedestal
                + (ion_temperature_centre - ion_temperature_pedestal)
                * (1 - (r_core / pedestal_radius) ** ion_temperature_beta)
                ** ion_temperature_peaking_factor
            ),
            (
                ion_temperature_separatrix
                + (ion_temperature_pedestal - ion_temperature_separatrix)
                * (minor_radius - r)
                / (minor_radius - pedestal_radius)
            ),
        )
    else:
        raise ValueError(f'Mode {mode} not in available options ["L", "H", "A"]')
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
        shafranov_factor: Shafranov factor (cm)
        minor_radius: Plasma minor radius (cm)
        major_radius: Plasma major radius (cm)
        triangularity: Plasma triangularity (dimensionless)
        elongation: Plasma elongation (dimensionless)

    Returns:
        (R, Z) coordinates (cm, cm)
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
