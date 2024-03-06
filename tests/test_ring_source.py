import numpy as np
import openmc
import pytest

from openmc_plasma_source import fusion_ring_source


def test_creation():
    my_source = fusion_ring_source(radius=1.0, z_placement=1.0)

    # Ensure it is of type openmc.IndependentSource
    assert isinstance(my_source, openmc.IndependentSource)

    # Ensure it has space, angle, and energy set
    assert isinstance(my_source.space, openmc.stats.CylindricalIndependent)
    assert isinstance(my_source.angle, openmc.stats.Isotropic)
    assert isinstance(my_source.energy, openmc.stats.univariate.Normal)


@pytest.mark.parametrize("radius", [1, 5.6, 1e5, 7.0])
def test_radius(radius):
    # should allow any positive float
    my_source = fusion_ring_source(radius=radius)
    assert my_source.radius == radius


@pytest.mark.parametrize("radius", [-1.0, "hello world", [1e5]])
def test_bad_radius(radius):
    # should reject any negative float or anything not convertible to float
    with pytest.raises(ValueError):
        fusion_ring_source(radius=radius)


@pytest.mark.parametrize("angles", [(1, 2), (0.0, np.pi), (np.pi, 0.0)])
def test_angles(angles):
    # Should allow any tuple of length 2 with contents convertible to float
    my_source = fusion_ring_source(radius=1.0, angles=angles)
    assert np.array_equal(my_source.angles, angles)


@pytest.mark.parametrize("angles", [(1.0,), [1, 2], 5, "ab", ("a", "b")])
def test_bad_angles(angles):
    # Should reject iterables of length != 2, anything non tuple, and anything
    # that can't convert to float
    with pytest.raises(ValueError):
        fusion_ring_source(radius=1.0, angles=angles)


@pytest.mark.parametrize("temperature", [20000.0, 1e4, 0.1, 25000])
def test_temperature(temperature):
    # Should accept any positive float
    my_source = fusion_ring_source(radius=1.0, temperature=temperature)
    assert my_source.temperature == temperature


@pytest.mark.parametrize("temperature", [-20000.0, "hello world", [10000]])
def test_bad_temperature(temperature):
    # Should reject negative floats and anything that isn't convertible to float
    with pytest.raises(ValueError):
        fusion_ring_source(radius=1.0, temperature=temperature)


@pytest.mark.parametrize("fuel", ["DT", "DD"])
def test_fuel(fuel):
    # Should accept either 'DD' or 'DT'
    my_source = fusion_ring_source(radius=1.0, fuel=fuel)
    assert my_source.fuel_type == fuel


@pytest.mark.parametrize("fuel", ["топливо", 5])
def test_wrong_fuel(fuel):
    # Should reject fuel types besides those listed in fuel_types.py
    with pytest.raises((KeyError, TypeError)):
        fusion_ring_source(radius=1.0, fuel=fuel)


@pytest.mark.parametrize("z", ["coucou", [5, 2.0]])
def test_wrong_z_placement(z):
    with pytest.raises((TypeError)):
        fusion_ring_source(radius=1.0, z_placement=z)
