from openmc_plasma_source import FusionPointSource

from openmc import Source
import pytest

def test_creation():
    my_source = FusionPointSource()
    assert isinstance(my_source, Source)

with pytest.raises(ValueError):
    FusionRingSource(radius=1, fuel='топливо')
