from openmc_plasma_source import FusionPointSource

from openmc import Source
import pytest


def test_creation():
    my_source = FusionPointSource()
    assert isinstance(my_source, Source)


def test_wrong_fuel():
    with pytest.raises(ValueError):
        FusionPointSource(fuel="топливо")
