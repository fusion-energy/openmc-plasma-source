from openmc_plasma_source import FusionRingSource

from openmc import Source


def test_creation():
    my_source = FusionRingSource(radius=1)
    assert isinstance(my_source, Source)
