"""fuel_types.py

Defines dictionary for determining mean energy and mass of reactants
for a given fusion fuel type.
"""

from param import Parameterized, Number


class Fuel(Parameterized):

    # mean energy, eV
    mean_energy = Number(None, bounds=(0, None), inclusive_bounds=(False, False))

    # mass of the reactants, AMU
    mass_of_reactants = Number(None, bounds=(0, None), inclusive_bounds=(False, False))

    def __init__(self, mean_energy, mass_of_reactants):
        self.mean_energy = mean_energy
        self.mass_of_reactants = mass_of_reactants


fuel_types = {
    "DD": Fuel(mean_energy=2450000.0, mass_of_reactants=4),
    "DT": Fuel(mean_energy=14080000.0, mass_of_reactants=5),
}
