from openmc_plasma_source.fuel_types import Fuel, fuel_types
import pytest


@pytest.mark.parametrize("energy,mass", [(2.5e7, 5), (15, 30)])
def test_fuel_with_correct_inputs(energy, mass):
    # Should accept any non-zero positive inputs to either variable
    fuel = Fuel(energy, mass)
    assert fuel.mean_energy == energy
    assert fuel.mass_of_reactants == mass


@pytest.mark.parametrize(
    "energy,mass", [(2.5e7, -5), (-12, 30), (1e7, 0), (0, 4), (-12, -12)]
)
def test_fuel_with_bad_inputs(energy, mass):
    # Should reject any negative numbers and zeros.
    with pytest.raises(ValueError):
        fuel = Fuel(energy, mass)


@pytest.mark.parametrize("fuel_type", ["DT", "DD"])
def test_fuel_types(fuel_type):
    # Should accept 'DD' and 'DT'
    assert isinstance(fuel_types[fuel_type], Fuel)


@pytest.mark.parametrize("fuel_type", ["dt", "dd", "Dt", "dD", "hello world", 5])
def test_incorrect_fuel_types(fuel_type):
    # Should reject everything except 'DT' and 'DD'
    with pytest.raises(KeyError):
        my_fuel = fuel_types[fuel_type]
