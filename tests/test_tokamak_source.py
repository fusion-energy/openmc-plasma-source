import numpy as np
import openmc
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from openmc_plasma_source import (
    tokamak_convert_a_alpha_to_R_Z,
    tokamak_ion_density,
    tokamak_ion_temperature,
    tokamak_source,
)
from openmc_plasma_source.tokamak_source import _toroidal_phi_grid

# Small mesh/grid params used across tests for speed
FAST_MESH = (20, 20)
FAST_GRID = 50


@pytest.fixture
def tokamak_args_dict():
    """Returns a dict of realistic inputs for tokamak_source"""
    args_dict = {
        "elongation": 1.557,
        "triangularity": 0.270,
        "major_radius": 9.06,
        "minor_radius": 2.92258,
        "pedestal_radius": 0.8 * 2.92258,
        "shafranov_factor": 0.44789,
        "ion_density_centre": 1.09e20,
        "ion_density_peaking_factor": 1,
        "ion_density_pedestal": 1.09e20,
        "ion_density_separatrix": 3e19,
        "ion_temperature_centre": 45.9,
        "ion_temperature_peaking_factor": 8.06,
        "ion_temperature_pedestal": 6.09,
        "ion_temperature_separatrix": 0.1,
        "mode": "H",
        "ion_temperature_beta": 6,
        "mesh_resolution": FAST_MESH,
        "grid_density": FAST_GRID,
    }
    return args_dict


@pytest.fixture
def tokamak_source_example(tokamak_args_dict):
    """Returns a tokamak_source with realistic inputs"""
    return tokamak_source(**tokamak_args_dict)


def test_creation(tokamak_source_example):
    """Tests that tokamak_source returns an openmc.MeshSource"""
    assert isinstance(tokamak_source_example, openmc.MeshSource)


@pytest.mark.parametrize(
    "minor_radius,major_radius", [(3.0, 10.0), (3.0, 100), (3.0, 3.00001)]
)
def test_major_radius(tokamak_args_dict, minor_radius, major_radius):
    """Checks that tokamak_source creation accepts valid major radius"""
    tokamak_args_dict["minor_radius"] = minor_radius
    tokamak_args_dict["major_radius"] = major_radius
    tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize(
    "minor_radius,major_radius", [(3, 3), (3, 1), (3, -5), (3, "hello world")]
)
def test_bad_major_radius(tokamak_args_dict, minor_radius, major_radius):
    """Checks that tokamak_source creation rejects invalid major radius"""
    tokamak_args_dict["minor_radius"] = minor_radius
    tokamak_args_dict["major_radius"] = major_radius
    with pytest.raises((ValueError, TypeError)):
        tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize(
    "major_radius,minor_radius", [(10.0, 3.0), (10.0, 9.9), (10.0, 0.1)]
)
def test_minor_radius(tokamak_args_dict, major_radius, minor_radius):
    """Checks that tokamak_source creation accepts valid minor radius"""
    tokamak_args_dict["major_radius"] = major_radius
    tokamak_args_dict["minor_radius"] = minor_radius
    # Set shafranov factor to 0 and pedestal factor to 0.8*minor_radius for safety
    tokamak_args_dict["pedestal_radius"] = 0.8 * minor_radius
    tokamak_args_dict["shafranov_factor"] = 0.0
    tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize(
    "major_radius,minor_radius",
    [(10.0, 10.0), (10.0, 20.0), (10.0, 0), (10.0, -6), (10.0, "hello world")],
)
def test_bad_minor_radius(tokamak_args_dict, major_radius, minor_radius):
    """Checks that tokamak_source creation rejects invalid minor radius"""
    tokamak_args_dict["major_radius"] = major_radius
    tokamak_args_dict["minor_radius"] = minor_radius
    with pytest.raises((ValueError, TypeError)):
        tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize("elongation", [1.0, 1.667, 0.5, 20, 0.001])
def test_elongation(tokamak_args_dict, elongation):
    """Checks that tokamak_source creation accepts valid elongation"""
    tokamak_args_dict["elongation"] = elongation
    tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize("elongation", [0, -5, "hello world"])
def test_bad_elongation(tokamak_args_dict, elongation):
    """Checks that tokamak_source creation rejects invalid elongation"""
    tokamak_args_dict["elongation"] = elongation
    with pytest.raises((ValueError, TypeError)):
        tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize("triangularity", [0.0, 0.5, 0.9, 1.0, -0.5, -0.9, -1.0])
def test_triangularity(tokamak_args_dict, triangularity):
    """Checks that tokamak_source creation accepts valid triangularity"""
    tokamak_args_dict["triangularity"] = triangularity
    tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize("triangularity", [1.1, -1.1, 10, -10, "hello world"])
def test_bad_triangularity(tokamak_args_dict, triangularity):
    """Checks that tokamak_source creation rejects invalid triangularity"""
    tokamak_args_dict["triangularity"] = triangularity
    with pytest.raises((ValueError, TypeError)):
        tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize(
    "major_radius,minor_radius,shaf",
    [
        (10.0, 3.0, 0),
        (10.0, 3.0, 1.0),
        (10.0, 3.0, -1.0),
        (10.0, 3.0, 1.49),
        (10.0, 3.0, -1.49),
        (10.0, 5.0, 2.49),
        (10.0, 5.0, -2.49),
    ],
)
def test_shafranov_factor(tokamak_args_dict, major_radius, minor_radius, shaf):
    """Checks that tokamak_source creation accepts valid Shafranov factor"""
    tokamak_args_dict["major_radius"] = major_radius
    tokamak_args_dict["minor_radius"] = minor_radius
    tokamak_args_dict["pedestal_radius"] = 0.8 * minor_radius
    tokamak_args_dict["shafranov_factor"] = shaf
    tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize(
    "major_radius,minor_radius,shaf",
    [
        (10.0, 3.0, 3.0),
        (10.0, 3.0, -3.0),
        (10.0, 3.0, 1.5),
        (10.0, 3.0, -1.5),
        (10.0, 5.0, 2.5),
        (10.0, 5.0, -2.5),
    ],
)
def test_bad_shafranov_factor(tokamak_args_dict, major_radius, minor_radius, shaf):
    """Checks that tokamak_source creation rejects invalid Shafranov factor"""
    tokamak_args_dict["major_radius"] = major_radius
    tokamak_args_dict["minor_radius"] = minor_radius
    tokamak_args_dict["pedestal_radius"] = 0.8 * minor_radius
    tokamak_args_dict["shafranov_factor"] = shaf
    with pytest.raises((ValueError, TypeError)):
        tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize(
    "start_angle, rotation_angle",
    [
        (0, 1),
        (0, 2 * np.pi),
        (np.pi / 4, np.pi / 2),
        (-np.pi, np.pi),
        (3 * np.pi / 4, -np.pi / 4),
        (7 * np.pi / 4, np.pi / 2),  # crosses the 0 / 2*pi seam
    ],
)
def test_angles(tokamak_args_dict, start_angle, rotation_angle):
    """Checks that a valid sector produces a usable mesh with normalized strengths."""
    tokamak_args_dict["start_angle"] = start_angle
    tokamak_args_dict["rotation_angle"] = rotation_angle
    mesh_source = tokamak_source(**tokamak_args_dict)
    phi_grid = np.asarray(mesh_source.mesh.phi_grid)
    # Mesh edges must be increasing and within [0, 2*pi]
    assert np.all(np.diff(phi_grid) > 0)
    assert phi_grid[0] >= -1e-12
    assert phi_grid[-1] <= 2 * np.pi + 1e-9
    # Source strengths are normalized to sum to 1
    strengths = np.array([s.strength for s in mesh_source.sources.ravel()])
    assert np.isclose(strengths.sum(), 1.0)


@pytest.mark.parametrize(
    "start_angle, rotation_angle",
    [
        (0, np.pi / 2),
        (0, 2 * np.pi),
        (-np.pi, np.pi),
        (3 * np.pi / 4, -np.pi / 4),
        (7 * np.pi / 4, np.pi / 2),  # crosses the 0 / 2*pi seam
        (-np.pi / 4, np.pi / 2),  # crosses the seam
    ],
)
def test_toroidal_phi_grid(start_angle, rotation_angle):
    """The phi grid is valid and its active bins span exactly abs(rotation_angle)."""
    phi_grid, phi_fraction = _toroidal_phi_grid(start_angle, rotation_angle, n_phi=4)
    bin_widths = np.diff(phi_grid)
    # Valid CylindricalMesh edges
    assert np.all(bin_widths > 0)
    assert phi_grid[0] >= -1e-12
    assert phi_grid[-1] <= 2 * np.pi + 1e-9
    # Strength fractions sum to 1 and the active bins span the requested extent
    assert np.isclose(phi_fraction.sum(), 1.0)
    assert np.isclose(bin_widths[phi_fraction > 0].sum(), abs(rotation_angle))


@pytest.mark.parametrize("start_angle, rotation_angle", [(7 * np.pi / 4, np.pi / 2)])
def test_seam_crossing_sector_has_dead_bin(start_angle, rotation_angle):
    """A seam-crossing sector covers the full circle with a zero-strength bin."""
    phi_grid, phi_fraction = _toroidal_phi_grid(start_angle, rotation_angle, n_phi=4)
    assert np.isclose(phi_grid[0], 0.0)
    assert np.isclose(phi_grid[-1], 2 * np.pi)
    assert np.any(phi_fraction == 0.0)  # the dead bin


@pytest.mark.parametrize("start_angle", [3 * np.pi, -3 * np.pi, "hello", (0, 1)])
def test_bad_start_angle(tokamak_args_dict, start_angle):
    """Checks that invalid start_angle values are rejected"""
    tokamak_args_dict["start_angle"] = start_angle
    with pytest.raises((ValueError, TypeError)):
        tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize("rotation_angle", [3 * np.pi, -3 * np.pi, 0, "hello", (0, 1)])
def test_bad_rotation_angle(tokamak_args_dict, rotation_angle):
    """Checks that invalid rotation_angle values are rejected"""
    tokamak_args_dict["rotation_angle"] = rotation_angle
    with pytest.raises((ValueError, TypeError)):
        tokamak_source(**tokamak_args_dict)


@pytest.mark.parametrize("mesh_resolution", [(10, 1, 10), (10,), (10, 10, 10, 10)])
def test_bad_mesh_resolution(tokamak_args_dict, mesh_resolution):
    """mesh_resolution must be a 2-tuple (n_r, n_z); other lengths are rejected.

    In particular the old 3-tuple (n_r, n_phi, n_z) form should fail with a
    helpful message rather than silently mis-binning.
    """
    tokamak_args_dict["mesh_resolution"] = mesh_resolution
    with pytest.raises(ValueError, match="two values"):
        tokamak_source(**tokamak_args_dict)


def test_mesh_resolution_sets_single_phi_bin(tokamak_args_dict):
    """A 2-tuple (n_r, n_z) yields a mesh whose toroidal dimension is a single
    bin for a full rotation, and the requested r/z resolution."""
    tokamak_args_dict["mesh_resolution"] = (12, 8)
    tokamak_args_dict["rotation_angle"] = 2 * np.pi
    source = tokamak_source(**tokamak_args_dict)
    n_r, n_phi, n_z = source.mesh.dimension
    assert (n_r, n_phi, n_z) == (12, 1, 8)


def test_ion_density(tokamak_args_dict):
    # test with values of r that are within acceptable ranges.
    r = np.linspace(0.0, tokamak_args_dict["minor_radius"], 100)
    density = tokamak_ion_density(
        r=r,
        mode="L",
        ion_density_centre=tokamak_args_dict["ion_density_centre"],
        ion_density_peaking_factor=tokamak_args_dict["ion_density_peaking_factor"],
        ion_density_pedestal=tokamak_args_dict["ion_density_pedestal"],
        minor_radius=tokamak_args_dict["minor_radius"],
        pedestal_radius=tokamak_args_dict["pedestal_radius"],
        ion_density_separatrix=tokamak_args_dict["ion_density_separatrix"],
    )
    assert isinstance(r, np.ndarray)
    assert len(density) == len(r)
    assert np.all(np.isfinite(density))


def test_bad_ion_density(tokamak_args_dict):
    # It should fail if given a negative r
    with pytest.raises(ValueError) as excinfo:
        r = [0, 5, -6]
        tokamak_ion_density(
            r=r,
            mode="L",
            ion_density_centre=tokamak_args_dict["ion_density_centre"],
            ion_density_peaking_factor=tokamak_args_dict["ion_density_peaking_factor"],
            ion_density_pedestal=tokamak_args_dict["ion_density_pedestal"],
            minor_radius=tokamak_args_dict["minor_radius"],
            pedestal_radius=tokamak_args_dict["pedestal_radius"],
            ion_density_separatrix=tokamak_args_dict["ion_density_separatrix"],
        )
    assert "must not be negative" in str(excinfo.value)


def test_ion_temperature(tokamak_args_dict, tokamak_source_example):
    # test with values of r that are within acceptable ranges.
    r = np.linspace(0.0, 2.9, 100)
    temperature = tokamak_ion_temperature(
        r=r,
        mode=tokamak_args_dict["mode"],
        pedestal_radius=tokamak_args_dict["pedestal_radius"],
        ion_temperature_pedestal=tokamak_args_dict["ion_temperature_pedestal"],
        ion_temperature_centre=tokamak_args_dict["ion_temperature_centre"],
        ion_temperature_beta=tokamak_args_dict["ion_temperature_beta"],
        ion_temperature_peaking_factor=tokamak_args_dict[
            "ion_temperature_peaking_factor"
        ],
        ion_temperature_separatrix=tokamak_args_dict["ion_temperature_separatrix"],
        minor_radius=tokamak_args_dict["minor_radius"],
    )
    assert isinstance(temperature, np.ndarray)
    assert len(temperature) == len(r)
    assert np.all(np.isfinite(temperature))


def test_bad_ion_temperature(tokamak_args_dict):
    # It should fail if given a negative r
    with pytest.raises(ValueError) as excinfo:
        r = [0, 5, -6]
        tokamak_ion_temperature(
            r=r,
            mode=tokamak_args_dict["mode"],
            pedestal_radius=tokamak_args_dict["pedestal_radius"],
            ion_temperature_pedestal=tokamak_args_dict["ion_temperature_pedestal"],
            ion_temperature_centre=tokamak_args_dict["ion_temperature_centre"],
            ion_temperature_beta=tokamak_args_dict["ion_temperature_beta"],
            ion_temperature_peaking_factor=tokamak_args_dict[
                "ion_temperature_peaking_factor"
            ],
            ion_temperature_separatrix=tokamak_args_dict["ion_temperature_separatrix"],
            minor_radius=tokamak_args_dict["minor_radius"],
        )
    assert "must not be negative" in str(excinfo.value)


def test_convert_a_alpha_to_R_Z(tokamak_args_dict):
    # Similar to  test_source_locations_are_within_correct_range
    # Rather than going in detail, simply tests validity of inputs and outputs
    # Test with suitable values for a and alpha
    a = np.linspace(0.0, 2.9, 100)
    alpha = np.linspace(0.0, 2 * np.pi, 100)
    R, Z = tokamak_convert_a_alpha_to_R_Z(
        a=a,
        alpha=alpha,
        shafranov_factor=tokamak_args_dict["shafranov_factor"],
        minor_radius=tokamak_args_dict["minor_radius"],
        major_radius=tokamak_args_dict["major_radius"],
        triangularity=tokamak_args_dict["triangularity"],
        elongation=tokamak_args_dict["elongation"],
    )
    assert isinstance(R, np.ndarray)
    assert isinstance(Z, np.ndarray)
    assert len(R) == len(a)
    assert len(Z) == len(a)
    assert np.all(np.isfinite(R))
    assert np.all(np.isfinite(Z))


def test_bad_convert_a_alpha_to_R_Z(tokamak_args_dict):
    # Repeat test_convert_a_alpha_to_R_Z, but show that negative a breaks it
    a = np.linspace(0.0, 2.9, 100)
    alpha = np.linspace(0.0, 2 * np.pi, 100)
    with pytest.raises(ValueError) as excinfo:
        tokamak_convert_a_alpha_to_R_Z(
            a=-a,
            alpha=alpha,
            shafranov_factor=tokamak_args_dict["shafranov_factor"],
            minor_radius=tokamak_args_dict["minor_radius"],
            major_radius=tokamak_args_dict["major_radius"],
            triangularity=tokamak_args_dict["triangularity"],
            elongation=tokamak_args_dict["elongation"],
        )
    assert "must not be negative" in str(excinfo.value)


@st.composite
def tokamak_source_strategy(draw):
    """Defines a hypothesis strategy that automatically generates a tokamak_source.
    Geometry attributes are varied, while plasma attributes are fixed.
    """
    # Used to avoid generation of inappropriate float values
    finites = {
        "allow_nan": False,
        "allow_infinity": False,
        "allow_subnormal": False,
    }

    # Specify the base strategies for each geometry input
    major_radius = draw(st.floats(min_value=1e-5, max_value=100.0, **finites))

    minor_radius = draw(
        st.floats(
            min_value=1e-5 * major_radius,
            max_value=np.nextafter(major_radius, major_radius - 1),
            **finites,
        )
    )

    pedestal_radius = draw(
        st.floats(
            min_value=0.8 * minor_radius,
            max_value=np.nextafter(minor_radius, minor_radius - 1),
            **finites,
        )
    )

    elongation = draw(st.floats(min_value=1e-5, max_value=10.0, **finites))

    triangularity = draw(
        st.floats(
            min_value=np.nextafter(-1.0, +1), max_value=np.nextafter(1.0, -1), **finites
        )
    )

    shafranov_factor = draw(
        st.floats(
            min_value=np.nextafter(-0.5 * minor_radius, +1),
            max_value=np.nextafter(0.5 * minor_radius, -1),
            **finites,
        )
    )

    return tokamak_source(
        elongation=elongation,
        triangularity=triangularity,
        major_radius=major_radius,
        minor_radius=minor_radius,
        pedestal_radius=pedestal_radius,
        shafranov_factor=shafranov_factor,
        ion_density_centre=1.09e20,
        ion_density_peaking_factor=1,
        ion_density_pedestal=1.01e20,
        ion_density_separatrix=3e19,
        ion_temperature_centre=45.9,
        ion_temperature_peaking_factor=8.06,
        ion_temperature_pedestal=6.09,
        ion_temperature_separatrix=0.1,
        mode="H",
        ion_temperature_beta=6,
        mesh_resolution=(10, 10),
        grid_density=30,
    ), {
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "elongation": elongation,
        "triangularity": triangularity,
    }


@given(tokamak_source=tokamak_source_strategy())
@settings(max_examples=30, suppress_health_check=(HealthCheck.too_slow,))
def test_strengths_are_normalised(tokamak_source):
    """Tests that the sum of the strengths attribute is equal to 1"""
    local_strength = 0
    mesh_source = tokamak_source[0]
    for source in mesh_source.sources.flat:
        local_strength = local_strength + source.strength
    assert pytest.approx(local_strength) == 1


@given(tokamak_source=tokamak_source_strategy())
@settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
def test_source_locations_are_within_correct_range(tokamak_source):
    """Tests that the mesh bounds encompass the plasma cross-section.

    The mesh R bounds should contain [R0-a, R0+a] and the Z bounds should
    contain [-kappa*a, kappa*a], which is the extent of the last closed
    magnetic surface.
    """
    mesh_source = tokamak_source[0]
    R_0 = tokamak_source[1]["major_radius"]
    A = tokamak_source[1]["minor_radius"]
    El = tokamak_source[1]["elongation"]

    r_grid = np.asarray(mesh_source.mesh.r_grid)
    z_grid = np.asarray(mesh_source.mesh.z_grid)

    # Mesh R bounds encompass the LCMS
    assert r_grid[0] <= R_0 - A or np.isclose(r_grid[0], R_0 - A)
    assert r_grid[-1] >= R_0 + A or np.isclose(r_grid[-1], R_0 + A)

    # Mesh Z bounds encompass the LCMS
    assert z_grid[0] <= -El * A or np.isclose(z_grid[0], -El * A)
    assert z_grid[-1] >= El * A or np.isclose(z_grid[-1], El * A)


def _uniform_args(**overrides):
    """Flat density and temperature (L mode, zero peaking), so the emission
    per unit volume is uniform and the source is pure geometry."""
    args = {
        "major_radius": 9.06,
        "minor_radius": 2.92258,
        "elongation": 1.0,
        "triangularity": 0.0,
        "shafranov_factor": 0.0,
        "mode": "L",
        "ion_density_centre": 1.0e20,
        "ion_density_peaking_factor": 0,
        "ion_density_pedestal": 1.0e20,
        "ion_density_separatrix": 1.0e19,
        "ion_temperature_centre": 20e3,
        "ion_temperature_peaking_factor": 0,
        "ion_temperature_beta": 2,
        "ion_temperature_pedestal": 5e3,
        "ion_temperature_separatrix": 100,
        "pedestal_radius": 0.8 * 2.92258,
        "mesh_resolution": (40, 40),
        "grid_density": 200,
    }
    args.update(overrides)
    return args


def _strength_moments(mesh_source):
    """Strength-weighted <R> and <Z^2> over mesh cell centres."""
    mesh = mesh_source.mesh
    # MeshSource stores sources flattened in mesh index (Fortran) order
    strengths = np.array([src.strength for src in mesh_source.sources]).reshape(
        mesh.dimension, order="F"
    )
    strengths = strengths.sum(axis=1)
    r_grid = np.asarray(mesh.r_grid)
    z_grid = np.asarray(mesh.z_grid)
    r_centers = 0.5 * (r_grid[:-1] + r_grid[1:])
    z_centers = 0.5 * (z_grid[:-1] + z_grid[1:])
    mean_r = np.average(r_centers, weights=strengths.sum(axis=1))
    mean_z = np.average(z_centers, weights=strengths.sum(axis=0))
    mean_z2 = np.average(z_centers**2, weights=strengths.sum(axis=0))
    return mean_r, mean_z, mean_z2, z_grid[1] - z_grid[0]


def test_strengths_are_volume_weighted():
    """Source strengths must include the plasma volume element R * |J|.

    For uniform emission in a circular torus, Pappus gives the
    strength-weighted mean radius exactly: <R> = R0 + a^2 / (4 R0). Without
    the toroidal R factor it would be R0. See "Tokamak D-T neutron source
    models for different plasma physics confinement modes", C. Fausser et al.,
    Fusion Engineering and Design, 2012.
    """
    args = _uniform_args()
    R0, a = args["major_radius"], args["minor_radius"]
    mean_r, mean_z, _, _ = _strength_moments(tokamak_source(**args))
    expected = R0 + a**2 / (4 * R0)
    assert mean_r == pytest.approx(expected, abs=0.02 * (expected - R0))
    # up-down symmetric plasma, so the emission is centred on the midplane
    assert mean_z == pytest.approx(0.0, abs=1e-3 * a)


@pytest.mark.parametrize(
    "elongation, triangularity, shafranov_factor",
    [(1.557, 0.27, 0.0), (1.557, 0.27, 0.44789), (1.8, 0.5, -0.9), (1.3, -0.4, 1.2)],
)
def test_shaped_strengths_match_boundary_integrals(
    elongation, triangularity, shafranov_factor
):
    """Uniform emission in a shaped plasma, checked against the boundary alone.

    Green's theorem turns the area integrals into loop integrals over the last
    closed surface, which the Shafranov shift does not move:
        int R dA = loop R^2/2 dZ,  int R^2 dA = loop R^3/3 dZ,
        int R Z^2 dA = loop R^2 Z^2/2 dZ
    so the reference never touches the (a, alpha) Jacobian used by the code.
    """
    args = _uniform_args(
        elongation=elongation,
        triangularity=triangularity,
        shafranov_factor=shafranov_factor,
    )
    R0, a = args["major_radius"], args["minor_radius"]

    # periodic integrands, so the uniform rule converges spectrally
    t = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    R = R0 + a * np.cos(t + triangularity * np.sin(t))
    Z = elongation * a * np.sin(t)
    dZ = elongation * a * np.cos(t)
    I1 = np.mean(R**2 / 2 * dZ)
    I2 = np.mean(R**3 / 3 * dZ)
    I3 = np.mean(R**2 * Z**2 / 2 * dZ)

    mean_r, mean_z, mean_z2, dz = _strength_moments(tokamak_source(**args))
    assert mean_z == pytest.approx(0.0, abs=1e-3 * a)
    expected_r = I2 / I1
    assert mean_r == pytest.approx(
        expected_r, abs=0.05 * abs(expected_r - R0) + 1e-3 * a
    )
    # cell-centre binning adds dz^2 / 12 to <Z^2>, small next to the tolerance here
    assert mean_z2 == pytest.approx(I3 / I1 + dz**2 / 12, rel=5e-3)
