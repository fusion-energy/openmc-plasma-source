from openmc_plasma_source import TokamakSource
from openmc import Source
import math

import pytest


@pytest.fixture
def tokamak_input():
    return {
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
    }


@pytest.fixture
def my_source(tokamak_input):
    return TokamakSource(**tokamak_input)


def test_creation(my_source):
    for source in my_source.sources:
        assert isinstance(source, Source)


def test_strengths_are_normalised(my_source):
    """Tests that the sum of the strengths attribute is equal to"""
    assert pytest.approx(sum(my_source.strengths), 1)


def test_source_locations_are_within_correct_range(tokamak_input, my_source):
    """Tests that each source has RZ locations within the expected range.
    Note that this test is not wholly accurate, as it only ensures that
    R and Z fall within a box defined by the lower-left point (R_min,Z_min)
    and upper-right point (R_max,Z_max), and does not account for the
    actual expected plasma geometry.
    """
    R_0 = tokamak_input["major_radius"]
    A = tokamak_input["minor_radius"]
    El = tokamak_input["elongation"]
    delta = tokamak_input["triangularity"]
    def get_R(alpha):
        return  R_0 + A * math.cos(alpha + delta * math.sin(alpha))
    for source in my_source.sources:
        R, Z = source.space.r.x[0], source.space.z.x[0]
        # First test that the point is contained with a simple box with
        # lower left (r_min,-z_max) and upper right (r_max,z_max)
        assert R >= R_0 - A
        assert R <= R_0 + A
        assert abs(Z) <= A * El
        # For a given Z, we can determine the two values of alpha where
        # where a = minor_radius, and from there determine the upper and
        # lower bounds for R.
        alpha_1 = math.asin(abs(Z) / (El * A))
        alpha_2 = math.pi - alpha_1
        R_max, R_min = get_R(alpha_1), get_R(alpha_2)
        assert R_max <= R_0 + A
        assert R_min >= R_0 - A
        assert R <= R_max
        assert R >= R_min


def test_angles(tokamak_input):
    """Checks that custom angles can be set"""
    alt_input = tokamak_input.copy()
    alt_input["angles"] = (0, 1)
    alt_source = TokamakSource(**alt_input)
    assert alt_source.angles == (0, 1)
    for source in alt_source.sources:
        assert (source.space.phi.a, source.space.phi.b) == (0, 1)
