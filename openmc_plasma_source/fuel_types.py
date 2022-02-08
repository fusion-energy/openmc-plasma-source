"""fuel_types.py

Defines dictionary for determining mean energy and mass of reactants
for a given fusion fuel type.
"""

from dataclasses import dataclass


@dataclass
class Fuel:
    mean_energy: float  # mean energy, eV
    mass_of_reactants: float  # mass of the reactants, AMU


fuel_types = {
    "DD": Fuel(mean_energy=2450000.0, mass_of_reactants=4),
    "DT": Fuel(mean_energy=14080000.0, mass_of_reactants=5),
}
