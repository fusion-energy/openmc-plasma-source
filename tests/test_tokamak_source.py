from openmc_plasma_source import TokamakSource

from openmc import Source

import pytest


def test_creation():
    my_source = TokamakSource(
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
    for source in my_source.make_openmc_sources():
        assert isinstance(source, Source)


def test_strengths_are_normalised():
    """Tests that the sum of the strengths attribute is equal to
    """
    my_source = TokamakSource(
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
    assert pytest.approx(sum(my_source.strengths), 1)


def test_angles():
    """Checks that custom angles can be set
    """
    my_source = TokamakSource(
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
        ion_temperature_beta=6,
        angles=(0, 1)
    )
