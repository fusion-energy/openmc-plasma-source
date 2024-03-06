import pytest

from openmc_plasma_source import get_neutron_energy_distribution


@pytest.mark.parametrize("temperature, fuel", [
    (2e3, {'D': 1.}),
    (2e3, {'T': 1.}),
    (2e3, {'T': 0.5, 'D': 0.5}),
    (2e3, {'T': 0.2, 'D': 0.8}),
])
def test_fuel_with_correct_inputs(temperature, fuel):
    # Should accept any non-zero positive inputs to either variable
    get_neutron_energy_distribution(temperature, fuel)


@pytest.mark.parametrize("temperature, fuel", [
    (2e3, {'D': 1.1}),
    (2e3, {'T': 0.9}),
    (2e3, {'T': -0.5, 'D': 0.5}),
    (2e3, {'T': -0.2, 'D': -0.8}),
])
def test_fuel_with_bad_inputs(temperature, fuel):
    # Should reject any negative numbers and zeros.
    with pytest.raises(ValueError):
        get_neutron_energy_distribution(temperature, fuel)


@pytest.mark.parametrize("temperature, fuel", [
    (2e3, {'DD': 1.1}),
    (2e3, {'DT': 0.9}),
    (2e3, {'He3': -0.5, 'D': 0.5}),
    (2e3, {1: -0.2, 'D': -0.8}),
])
def test_fuel_with_incorrect_isotopese(temperature, fuel):
    # Should reject anything which is not 'D' or 'T'.
    with pytest.raises(ValueError):
        get_neutron_energy_distribution(temperature, fuel)
