import pytest

from openmc_plasma_source import (
    get_neutron_energy_distribution,
    get_reactions_from_fuel,
)


@pytest.mark.parametrize(
    "temperature, fuel",
    [
        (2e3, {"D": 1.0}),
        (2e3, {"T": 1.0}),
        (2e3, {"T": 0.5, "D": 0.5}),
        (2e3, {"T": 0.2, "D": 0.8}),
    ],
)
def test_fuel_with_correct_inputs(temperature, fuel):
    # Should accept any non-zero positive inputs to either variable
    get_neutron_energy_distribution(temperature, fuel)


@pytest.mark.parametrize(
    "temperature, fuel",
    [
        (2e3, {"D": 1.1}),
        (2e3, {"T": 0.9}),
        (2e3, {"T": -0.5, "D": 0.5}),
        (2e3, {"T": -0.2, "D": -0.8}),
    ],
)
def test_fuel_with_bad_inputs(temperature, fuel):
    # Should reject any negative numbers and zeros.
    with pytest.raises(ValueError):
        get_neutron_energy_distribution(temperature, fuel)


@pytest.mark.parametrize(
    "temperature, fuel",
    [
        (2e3, {"DD": 1.1}),
        (2e3, {"DT": 0.9}),
        (2e3, {"He3": -0.5, "D": 0.5}),
        (2e3, {1: -0.2, "D": -0.8}),
    ],
)
def test_fuel_with_incorrect_isotopese(temperature, fuel):
    # Should reject anything which is not 'D' or 'T'.
    with pytest.raises(ValueError):
        get_neutron_energy_distribution(temperature, fuel)


@pytest.mark.parametrize(
    "fuel, expected_output",
    [
        ({"D": 1.0}, ["DD"]),
        ({"T": 1.0}, ["TT"]),
        ({"T": 0.5, "D": 0.5}, ["DT", "DD", "TT"]),
        ({"D": 0.2, "T": 0.8}, ["DT", "DD", "TT"]),
        ({"coucou": 0.2}, None),
    ],
)
def test_get_reactions_from_fuel(fuel, expected_output):
    """Test the get_reactions_from_fuel function"""
    if expected_output is None:
        with pytest.raises(ValueError, match=f"{fuel}"):
            get_reactions_from_fuel(fuel)
    else:
        assert get_reactions_from_fuel(fuel) == expected_output
