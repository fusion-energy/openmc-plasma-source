from openmc_plasma_source import FusionPointSource

import openmc
import pytest
import numpy as np


def test_creation():
    my_source = FusionPointSource()

    # Ensure it is of type openmc.Source
    assert isinstance(my_source, openmc.Source)

    # Ensure it has space, angle, and energy set
    assert isinstance(my_source.space, openmc.stats.Point)
    assert isinstance(my_source.angle, openmc.stats.Isotropic)
    assert isinstance(my_source.energy, openmc.stats.Muir)


def test_coordinate():
    # Should allow any iterable of length 3 with contents convertible to float
    success_states = [(1.0, 2.0, 3.0), [4, 5, 6], np.linspace(1.0, 3.0, 3)]
    for success_state in success_states:
        my_source = FusionPointSource(coordinate=success_state)
        assert np.all(np.equal(my_source.coordinate, success_state))


def test_bad_coordinate():
    # Should reject iterables of length != 3, anything non iterable, and anything
    # that can't convert to float
    fail_states = [(1, 2), [1, 2, 3, 4], 5, "abc"]
    for fail_state in fail_states:
        with pytest.raises(ValueError):
            FusionPointSource(coordinate=fail_state)


def test_temperature():
    # Should accept any positive float
    success_states = [20000.0, 1e4, 0.1, 25000]
    for success_state in success_states:
        my_source = FusionPointSource(temperature=success_state)
        assert my_source.temperature == success_state


def test_bad_temperature():
    # Should reject negative floats and anything that isn't convertible to float
    fail_states = [-20000.0, "hello world", [10000]]
    for fail_state in fail_states:
        with pytest.raises(ValueError):
            FusionPointSource(temperature=fail_state)


def test_fuel():
    # Should accept either 'DD' or 'DT'
    my_source = FusionPointSource(fuel_type="DT")
    assert my_source.fuel_type == "DT"
    my_source = FusionPointSource(fuel_type="DD")
    assert my_source.fuel_type == "DD"


def test_wrong_fuel():
    # Should reject fuel types besides those listed in fuel_types.py
    with pytest.raises(ValueError):
        FusionPointSource(fuel_type="топливо")
    # Should also reject non-strings
    with pytest.raises(ValueError):
        FusionPointSource(fuel_type=5)
