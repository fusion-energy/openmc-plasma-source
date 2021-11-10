from openmc_plasma_source import FusionRingSource

from openmc import Source
import pytest


def test_creation():
    my_source = FusionRingSource(radius=1, z_placement=1)
    assert isinstance(my_source, Source)


def test_wrong_fuel():
    with pytest.raises(ValueError):
        FusionRingSource(radius=1, fuel="топливо")
