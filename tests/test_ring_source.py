from openmc_plasma_source import FusionRingSource

import openmc
import pytest
import numpy as np


def test_creation():
    my_source = FusionRingSource(radius=1, z_placement=1)

    # Ensure it is of type openmc.Source
    assert isinstance(my_source, openmc.Source)

    # Ensure it has space, angle, and energy set
    assert isinstance(my_source.space, openmc.stats.CylindricalIndependent)
    assert isinstance(my_source.angle, openmc.stats.Isotropic)
    assert isinstance(my_source.energy, openmc.stats.Muir)


def test_radius():
    # should allow any positive float
    success_states = [1.0, 5.6, 1e5, 7]
    for success_state in success_states:
        my_source = FusionRingSource(radius=success_state)
        assert my_source.radius == success_state


def test_bad_radius():
    # should reject any negative float or anything not convertible to float
    failure_states = [-1.0, "hello world", [1e5]]
    for failure_state in failure_states:
        with pytest.raises(ValueError):
            my_source = FusionRingSource(radius=failure_state)


def test_angles():
    # Should allow any iterable of length 2 with contents convertible to float
    # If angles are given in reverse order, it should sort them automatically
    success_states = [(1.0, 2), [0, np.pi], (np.pi, 0)]
    for success_state in success_states:
        my_source = FusionRingSource(radius=1.0, angles=success_state)
        assert np.all(np.equal(my_source.angles, sorted(success_state)))
        assert my_source.angles[0] < my_source.angles[1]


def test_bad_angles():
    # Should reject iterables of length != 2, anything non iterable, and anything
    # that can't convert to float
    fail_states = [(1,), [1, 2, 3, 4], 5, "ab"]
    for fail_state in fail_states:
        with pytest.raises(ValueError):
            FusionRingSource(radius=1.0, angles=fail_state)


def test_temperature():
    # Should accept any positive float
    success_states = [20000.0, 1e4, 0.1, 25000]
    for success_state in success_states:
        my_source = FusionRingSource(radius=1.0, temperature=success_state)
        assert my_source.temperature == success_state


def test_bad_temperature():
    # Should reject negative floats and anything that isn't convertible to float
    fail_states = [-20000.0, "hello world", [10000]]
    for fail_state in fail_states:
        with pytest.raises(ValueError):
            FusionRingSource(radius=1.0, temperature=fail_state)


def test_fuel():
    # Should accept either 'DD' or 'DT'
    my_source = FusionRingSource(radius=1.0, fuel_type="DT")
    assert my_source.fuel_type == "DT"
    my_source = FusionRingSource(radius=1.0, fuel_type="DD")
    assert my_source.fuel_type == "DD"


def test_wrong_fuel():
    # Should reject fuel types besides those listed in fuel_types.py
    with pytest.raises(ValueError):
        FusionRingSource(radius=1.0, fuel_type="топливо")
    # Should also reject non-strings
    with pytest.raises(ValueError):
        FusionRingSource(radius=1.0, fuel_type=5)
