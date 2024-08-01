import numpy as np
import openmc
import pytest

from openmc_plasma_source import fusion_point_source


def test_creation():
    my_source = fusion_point_source()

    # Ensure it is of type openmc.IndependentSource
    for source in my_source:
        assert isinstance(source, openmc.IndependentSource)

        # Ensure it has space, angle, and energy set
        assert isinstance(source.space, openmc.stats.Point)
        assert isinstance(source.angle, openmc.stats.Isotropic)
        assert (
            isinstance(source.energy, openmc.stats.univariate.Normal)
            or isinstance(source.energy, openmc.stats.univariate.Tabular)
            or isinstance(source.energy, openmc.stats.Mixture)
        )


@pytest.mark.parametrize(
    "coordinate", [(1, 2, 3), (4.0, 5.0, 6.0), tuple(np.linspace(1.0, 3.0, 3))]
)
def test_coordinate(coordinate):
    # Should allow any tuple of length 3 containing numbers
    fusion_point_source(coordinate=coordinate)


@pytest.mark.parametrize(
    "coordinate", [(1.0, 2.0), [3, 4, 5], 5, "abc", ("a", "b", "c")]
)
def test_bad_coordinate(coordinate):
    # Should reject iterables of length != 3, anything non-tuple, and anything
    # that can't convert to float
    with pytest.raises(ValueError):
        fusion_point_source(coordinate=coordinate)


@pytest.mark.parametrize("temperature", [20000, 1e4, 0.1, 25000.0])
def test_temperature(temperature):
    # Should accept any positive float
    fusion_point_source(temperature=temperature)


@pytest.mark.parametrize("temperature", [-20000.0, "hello world", [10000]])
def test_bad_temperature(temperature):
    # Should reject negative floats and anything that isn't convertible to float
    with pytest.raises(ValueError):
        fusion_point_source(temperature=temperature)


@pytest.mark.parametrize("fuel", [{"D": 0.5, "T": 0.5}, {"D": 1.0}, {"T": 1.0}])
def test_fuel(fuel):
    # Should accept either 'DD' or 'DT'
    fusion_point_source(fuel=fuel)


@pytest.mark.parametrize("fuel", [{"топливо": 1.0}])
def test_wrong_fuel(fuel):
    # Should reject fuel types besides those listed in fuel_types.py
    with pytest.raises(ValueError):
        fusion_point_source(fuel=fuel)
