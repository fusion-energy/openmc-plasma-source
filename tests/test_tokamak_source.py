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

# Small mesh/grid params used across tests for speed
FAST_MESH = (20, 1, 20)
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


@pytest.mark.parametrize("angles", [(0, 1), (0, 2 * np.pi), (np.pi / 4, 3 * np.pi / 4)])
def test_angles(tokamak_args_dict, angles):
    """Checks that custom angles are reflected in the mesh phi_grid"""
    tokamak_args_dict["angles"] = angles
    mesh_source = tokamak_source(**tokamak_args_dict)
    phi_grid = mesh_source.mesh.phi_grid
    assert np.isclose(phi_grid[0], angles[0])
    assert np.isclose(phi_grid[-1], angles[1])


@pytest.mark.parametrize(
    "angles",
    [
        (0, 1, 2),
        -5,
        ("hello", "world"),
        (-np.pi, np.pi),
        (3 * np.pi / 4, -np.pi / 4),
    ],
)
def test_bad_angles(tokamak_args_dict, angles):
    """Checks that invalid angles are rejected"""
    tokamak_args_dict["angles"] = angles
    with pytest.raises((ValueError, TypeError)):
        tokamak_source(**tokamak_args_dict)


def test_ion_density(tokamak_args_dict):
    # test with values of r that are within acceptable ranges.
    r = np.linspace(0.0, tokamak_args_dict["minor_radius"], 100)
    density = tokamak_ion_density(
        r=r,
        mode="L",
        ion_density_centre=tokamak_args_dict["ion_density_centre"],
        ion_density_peaking_factor=tokamak_args_dict["ion_density_peaking_factor"],
        ion_density_pedestal=tokamak_args_dict["ion_density_pedestal"],
        major_radius=tokamak_args_dict["major_radius"],
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
            major_radius=tokamak_args_dict["major_radius"],
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
        major_radius=tokamak_args_dict["major_radius"],
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
            major_radius=tokamak_args_dict["major_radius"],
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
        mesh_resolution=(10, 1, 10),
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
