
def get_muir_paramters(reactants: str):
    
    if reactants == "DD":
        mean_energy=2450000.0
        mass_of_reactants=4
    elif reactants == 'DT:
        mean_energy=14080000.0
        mass_of_reactants=5
    else:
        raise ValueError(f'Only DD and DT fuel reactants are supported. Not {reactant}')

    return mean_energy, mass_of_reactants