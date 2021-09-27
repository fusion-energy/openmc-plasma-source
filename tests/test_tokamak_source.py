from openmc_plasma_source import TokamkaSource

from openmc import Source


def test_creation():
    my_source = TokamkaSource()
    for source in my_source:
        assert isinstance(source, Source)
