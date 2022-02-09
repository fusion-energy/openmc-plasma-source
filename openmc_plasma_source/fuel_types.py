"""fuel_types.py

Defines dictionary for determining mean energy and mass of reactants
for a given fusion fuel type.
"""

import proper_tea as pt


class Fuel:
    mean_energy = pt.positive_float(allow_zero=False)  # mean energy, eV
    mass_of_reactants = pt.positive(allow_zero=False)  # mass of the reactants, AMU

    def __init__(self, mean_energy, mass_of_reactants):
        self.mean_energy = mean_energy
        self.mass_of_reactants = mass_of_reactants


fuel_types = {
    "DD": Fuel(mean_energy=2450000.0, mass_of_reactants=4),
    "DT": Fuel(mean_energy=14080000.0, mass_of_reactants=5),
}
