import NeSST as nst
import numpy as np
import openmc


def get_neutron_energy_distribution(
    ion_temperature: float,
    fuel: dict,
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

    ion_temperature_kev = ion_temperature / 1e3  # convert eV to keV

    sum_fuel_isotopes = sum(fuel.values())
    if sum_fuel_isotopes > 1.0:
        raise ValueError(
            f"isotope fractions within the fuel must sum to be below 1. Not {sum_fuel_isotopes}"
        )

    if sum_fuel_isotopes < 0.0:
        raise ValueError(f"isotope must sum to be above 0. Not {sum_fuel_isotopes}")

    for k, v in fuel.items():
        if k not in ["D", "T"]:
            raise ValueError(
                f'Fuel dictionary keys must be either "D" or "T" not "{k}".'
            )
        if v < 0:
            raise ValueError(f'Fuel dictionary values must be above 0 not "{k}".')
        if v > 1:
            raise ValueError(f'Fuel dictionary values must be below 1 not "{k}".')

    # 1.0 neutron yield, all reactions scaled by this value
    num_of_vals = 100
    # single grid for TT neutrons
    E_pspec = np.linspace(0, 12, num_of_vals)  # E_pspec is exspected in MeV units

    DTmean, DTvar = nst.DTprimspecmoments(ion_temperature_kev)
    print('DTmean', DTmean)
    DDmean, DDvar = nst.DDprimspecmoments(ion_temperature_kev)

    if ["D", "T"] == sorted(set(fuel.keys())):

        strength_DT = 1.0
        strength_DD = nst.yield_from_dt_yield_ratio(
            "dd", strength_DT, ion_temperature_kev, fuel["D"], fuel["T"]
        )
        strength_TT = nst.yield_from_dt_yield_ratio(
            "tt", strength_DT, ion_temperature_kev, fuel["D"], fuel["T"]
        )

        dNdE_TT = strength_TT * nst.dNdE_TT(E_pspec, ion_temperature_kev)
        tt_source = openmc.stats.Tabular(E_pspec * 1e6, dNdE_TT)
        dd_source = openmc.stats.muir(e0=2.5e6, m_rat=4, kt=ion_temperature)
        dt_source = openmc.stats.muir(e0=14.06e6, m_rat=5, kt=ion_temperature)
        # todo look into combining distributions openmc.data.combine_distributions()
        return [tt_source, dd_source, dt_source], [strength_TT, strength_DD, strength_DT]

    elif ["D"] == sorted(set(fuel.keys())):

        strength_DD = 1.0
        dd_source = openmc.stats.muir(e0=2.5e6, m_rat=4, kt=ion_temperature)
        return [dd_source], [strength_DD]

    elif ["T"] == sorted(set(fuel.keys())):

        strength_TT = 1.0
        dNdE_TT = strength_TT * nst.dNdE_TT(E_pspec, ion_temperature_kev)
        tt_source = openmc.stats.Tabular(E_pspec * 1e6, dNdE_TT)
        return [tt_source], [strength_TT]

