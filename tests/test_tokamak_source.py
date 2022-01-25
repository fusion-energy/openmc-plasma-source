from openmc_plasma_source import TokamakSource
from openmc import Source
import numpy as np

import pytest
from hypothesis import given, settings, assume, strategies as st

finites = {"allow_nan": False, "allow_infinity": False, "allow_subnormal": False}


@pytest.fixture(scope="module")
def tokamak_plasma_input():
    return {
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
    }


@pytest.fixture(scope="module")
def tokamak_geometry_input():
    return {
        "elongation": 1.557,
        "triangularity": 0.270,
        "major_radius": 9.06,
        "minor_radius": 2.92258,
        "pedestal_radius": 0.8 * 2.92258,
        "shafranov_factor": 0.44789,
    }


@pytest.fixture(scope="module")
def tokamak_input(tokamak_plasma_input, tokamak_geometry_input):
    return {**tokamak_plasma_input, **tokamak_geometry_input}


@pytest.fixture(scope="function")
def tokamak_source(tokamak_input):
    return TokamakSource(**tokamak_input)


@st.composite
def tokamak_geometry(draw):
    minor_radius = draw(st.floats(min_value=0.0, max_value=100.0, **finites))
    major_radius = draw(st.floats(min_value=0.0, max_value=100.0, **finites))
    pedestal_radius = draw(st.floats(min_value=0.0, max_value=100.0, **finites))
    elongation = draw(st.floats(min_value=1.0, max_value=10.0, **finites))
    triangularity = draw(st.floats(min_value=0.0, max_value=1.0, **finites))
    shafranov_factor = draw(st.floats(min_value=0.0, max_value=1.0, **finites))
    assume(major_radius > minor_radius)
    assume(minor_radius > pedestal_radius)
    assume(minor_radius > shafranov_factor)
    return {
        "elongation": elongation,
        "triangularity": triangularity,
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "pedestal_radius": pedestal_radius,
        "shafranov_factor": shafranov_factor,
    }


def test_creation(tokamak_source):
    for source in tokamak_source.sources:
        assert isinstance(source, Source)


def test_strengths_are_normalised(tokamak_source):
    """Tests that the sum of the strengths attribute is equal to"""
    assert pytest.approx(sum(tokamak_source.strengths), 1)


@given(geometry=tokamak_geometry())
@settings(max_examples=100,deadline=None)
def test_source_locations_are_within_correct_range(geometry, tokamak_plasma_input):
    """Tests that each source has RZ locations within the expected range.
    Note that this test is not wholly accurate, as it only ensures that
    R and Z fall within a box defined by the lower-left point (R_min,Z_min)
    and upper-right point (R_max,Z_max), and does not account for the
    actual expected plasma geometry.
    """
    tokamak_source = TokamakSource(**geometry, **tokamak_plasma_input)
    R_0 = geometry["major_radius"]
    A = geometry["minor_radius"]
    El = geometry["elongation"]
    delta = geometry["triangularity"]

    def get_R(alpha):
        return R_0 + A * np.cos(alpha + delta * np.sin(alpha))
    
    for source in tokamak_source.sources:
        R, Z = source.space.r.x[0], source.space.z.x[0]
        # First test that the point is contained with a simple box with
        # lower left (r_min,-z_max) and upper right (r_max,z_max)
        assert R > R_0 - A or np.isclose(R, R_0 - A)
        assert R < R_0 + A or np.isclose(R, R_0 + A)
        assert abs(Z) < A * El or np.isclose(abs(Z), A * El)
        # For a given Z, we can determine the two values of alpha where
        # where a = minor_radius, and from there determine the upper and
        # lower bounds for R.
        alpha_1 = np.arcsin(abs(Z) / (El * A))
        alpha_2 = np.pi - alpha_1
        R_max, R_min = get_R(alpha_1), get_R(alpha_2)
        assert R_max < R_0 + A or np.isclose(R_max, R_0 + A)
        assert R_min > R_0 - A or np.isclose(R_min, R_0 - A)
        assert R < R_max or np.isclose(R, R_max)
        assert R > R_min or np.isclose(R, R_min)


def test_angles(tokamak_input):
    """Checks that custom angles can be set"""
    alt_input = tokamak_input.copy()
    alt_input["angles"] = (0, 1)
    alt_source = TokamakSource(**alt_input)
    assert alt_source.angles == (0, 1)
    for source in alt_source.sources:
        assert (source.space.phi.a, source.space.phi.b) == (0, 1)
