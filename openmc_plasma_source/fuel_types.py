"""fuel_types.py

Defines dictionary for determining mean energy and mass of reactants
for a given fusion fuel type.
"""


class Fuel:
    def __init__(self, mean_energy, mass_of_reactants):
        self.mean_energy = mean_energy
        self.mass_of_reactants = mass_of_reactants

    @property
    def mean_energy(self):
        return self._mean_energy

    @mean_energy.setter
    def mean_energy(self, value):
        if value <= 0:
            raise (ValueError("mean_energy needs to be strictly positive"))
        self._mean_energy = value

    @property
    def mass_of_reactants(self):
        return self._mass_of_reactants

    @mass_of_reactants.setter
    def mass_of_reactants(self, value):
        if value <= 0:
            raise (ValueError("mass_of_reactants needs to be strictly positive"))
        self._mass_of_reactants = value


fuel_types = {
    "DD": Fuel(mean_energy=2450000.0, mass_of_reactants=4),
    "DT": Fuel(mean_energy=14080000.0, mass_of_reactants=5),
}
