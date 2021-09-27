from openmc_plasma_source import FusionPointSource

from openmc import Source


def test_creation():
    my_source = FusionPointSource()
    assert isinstance(my_source, Source)
