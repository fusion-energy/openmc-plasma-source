from openmc_plasma_source import Plasma

from openmc import Source


def test_creation():
    my_source = Plasma()
    for source in my_source:
        assert isinstance(source, Source)
