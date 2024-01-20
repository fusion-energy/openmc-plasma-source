import NeSST as nst
import numpy as np
import openmc

def get_neutron_energy_distribution(
        ion_temperature: float,
        fuel: dict = {'D':0.5, 'T':0.5},
) -> openmc.stats.Discrete:
    """Finds the energy distribution

    Parameters
    ----------
    ion_temperature : float
        temperature of plasma ions in eV
    fuel : dict
        isotopes as keys and atom fractions as values

    Returns
    -------
    openmc.stats.Discrete
        energy distribution
    """

    ion_temperature = ion_temperature / 1e3  # convert eV to keV

    sum_fuel_isotopes = sum(fuel.values())
    if sum_fuel_isotopes > 1.:
        raise ValueError(f'isotope fractions within the fuel must sum to be below 1. Not {sum_fuel_isotopes}')

    if sum_fuel_isotopes < 0.:
        raise ValueError(f'isotope must sum to be above 0. Not {sum_fuel_isotopes}')

    for k, v in fuel.dict:
        if k not in ['D', 'T']:
            raise ValueError(f'Fuel dictionary keys must be either "D" or "T" not "{k}".')
        if v < 0:
            raise ValueError(f'Fuel dictionary values must be above 0 not "{k}".')
        if v > 1:
            raise ValueError(f'Fuel dictionary values must be below 1 not "{k}".')

    #Set atomic fraction of D and T in scattering medium and source
    if 'D' in fuel.keys():
        nst.frac_D_default = fuel['D']
        max_energy_mev=5
    else:
        nst.frac_D_default = 0

    if 'T' in fuel.keys():
        nst.frac_T_default = fuel['T']
        max_energy_mev=12
    else:
        nst.frac_T_default = 0
    
    if 'T' in fuel.keys() and 'D' in fuel.keys():
        max_energy_mev=20

    # 1.0 neutron yield, all reactions scaled by this value
    num_of_vals = 500
    # single grid for DT, DD and TT grid
    E_pspec = np.linspace(0, max_energy_mev, num_of_vals)  # accepts MeV units

    dNdE_DT_DD_TT = np.zeros(num_of_vals)
    if 'D' in fuel.keys() and 'T' in fuel.keys():
        DTmean, DTvar = nst.DTprimspecmoments(ion_temperature)
        DDmean, DDvar = nst.DDprimspecmoments(ion_temperature)

        Y_DT = 1.0
        Y_DD = nst.yield_from_dt_yield_ratio("dd", Y_DT, ion_temperature)
        Y_TT = nst.yield_from_dt_yield_ratio("tt", Y_DT, ion_temperature)

        dNdE_DT = Y_DT * nst.Qb(E_pspec, DTmean, DTvar)  # Brysk shape i.e. Gaussian
        dNdE_DD = Y_DD * nst.Qb(E_pspec, DDmean, DDvar)  # Brysk shape i.e. Gaussian
        dNdE_TT = Y_TT * nst.dNdE_TT(E_pspec, ion_temperature)
        dNdE_DT_DD_TT= dNdE_DT + dNdE_DD + dNdE_TT

    if 'D' in fuel.keys() and 'T' not in fuel.keys():
        DTmean, DTvar = nst.DTprimspecmoments(ion_temperature)
        DDmean, DDvar = nst.DDprimspecmoments(ion_temperature)

        Y_DD = 1.0

        dNdE_DD = Y_DD * nst.Qb(E_pspec, DDmean, DDvar)  # Brysk shape i.e. Gaussian
        dNdE_DT_DD_TT= dNdE_DD

    if 'D' not in fuel.keys() and 'T' in fuel.keys():

        Y_TT = 1.0
        
        dNdE_TT = Y_TT * nst.dNdE_TT(E_pspec, ion_temperature)
        dNdE_DT_DD_TT= dNdE_TT

    return openmc.stats.Discrete(E_pspec * 1e6, dNdE_DT_DD_TT)
