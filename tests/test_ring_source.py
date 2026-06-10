import numpy as np
import openmc
import pytest

from openmc_plasma_source import fusion_ring_source


def test_creation():
    my_source = fusion_ring_source(radius=1.0, z_placement=1.0)

    # Ensure it is of type openmc.IndependentSource
    for source in my_source:
        assert isinstance(source, openmc.IndependentSource)

        # Ensure it has space, angle, and energy set
        assert isinstance(source.space, openmc.stats.CylindricalIndependent)
        assert isinstance(source.angle, openmc.stats.Isotropic)
        assert (
            isinstance(source.energy, openmc.stats.univariate.Normal)
            or isinstance(source.energy, openmc.stats.univariate.Tabular)
            or isinstance(source.energy, openmc.stats.Mixture)
        )


@pytest.mark.parametrize("radius", [1, 5.6, 1e5, 7.0])
def test_radius(radius):
    # should allow any positive float
    fusion_ring_source(radius=radius)


@pytest.mark.parametrize("radius", [-1.0, "hello world", [1e5]])
def test_bad_radius(radius):
    # should reject any negative float or anything not convertible to float
    with pytest.raises(ValueError):
        fusion_ring_source(radius=radius)


@pytest.mark.parametrize(
    "start_angle, rotation_angle",
    [(0.0, np.pi), (np.pi, np.pi), (-np.pi, np.pi), (np.pi, -np.pi), (0.0, 2 * np.pi)],
)
def test_angles(start_angle, rotation_angle):
    # Should allow any start_angle and rotation_angle within -2*pi and 2*pi
    fusion_ring_source(
        radius=1.0, start_angle=start_angle, rotation_angle=rotation_angle
    )


@pytest.mark.parametrize("start_angle", [3 * np.pi, -3 * np.pi, "a", [1, 2]])
def test_bad_start_angle(start_angle):
    # Should reject values outside -2*pi to 2*pi and anything not a float
    with pytest.raises(ValueError):
        fusion_ring_source(radius=1.0, start_angle=start_angle)


@pytest.mark.parametrize("rotation_angle", [3 * np.pi, -3 * np.pi, "a", [1, 2]])
def test_bad_rotation_angle(rotation_angle):
    # Should reject values outside -2*pi to 2*pi and anything not a float
    with pytest.raises(ValueError):
        fusion_ring_source(radius=1.0, rotation_angle=rotation_angle)


@pytest.mark.parametrize("temperature", [20000.0, 1e4, 0.1, 25000])
def test_temperature(temperature):
    # Should accept any positive float
    fusion_ring_source(radius=1.0, temperature=temperature)


@pytest.mark.parametrize("temperature", [-20000.0, "hello world", [10000]])
def test_bad_temperature(temperature):
    # Should reject negative floats and anything that isn't convertible to float
    with pytest.raises(ValueError):
        fusion_ring_source(radius=1.0, temperature=temperature)


@pytest.mark.parametrize("fuel", [{"D": 0.5, "T": 0.5}, {"D": 1.0}])
def test_fuel(fuel):
    # Should accept either 'DD' or 'DT'
    fusion_ring_source(radius=1.0, fuel=fuel)


@pytest.mark.parametrize("fuel", [{"топливо": 1.0}])
def test_wrong_fuel(fuel):
    # Should reject fuel types besides those listed in fuel_types.py
    with pytest.raises(ValueError):
        fusion_ring_source(radius=1.0, fuel=fuel)


@pytest.mark.parametrize("z", ["coucou", [5, 2.0]])
def test_wrong_z_placement(z):
    with pytest.raises((TypeError)):
        fusion_ring_source(radius=1.0, z_placement=z)
