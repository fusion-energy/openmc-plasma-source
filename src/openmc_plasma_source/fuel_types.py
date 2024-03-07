import NeSST as nst
import numpy as np
import openmc


def get_neutron_energy_distribution(
    ion_temperature: float,
    fuel: dict,
) -> openmc.stats.Discrete:
    """Finds the energy distribution and their relative strengths.

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
        raise ValueError(
            f"isotope fractions must sum to be above 0. Not {sum_fuel_isotopes}"
        )

    if sum_fuel_isotopes != 1.0:
        raise ValueError(f"isotope fractions must sum to 1. Not {sum_fuel_isotopes}")

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
    DDmean, DDvar = nst.DDprimspecmoments(ion_temperature_kev)

    if ["D", "T"] == sorted(set(fuel.keys())):
        strength_DT = 1.0
        strength_DD = nst.yield_from_dt_yield_ratio(
            "dd", strength_DT, ion_temperature_kev, fuel["D"], fuel["T"]
        )
        strength_TT = nst.yield_from_dt_yield_ratio(
            "tt", strength_DT, ion_temperature_kev, fuel["D"], fuel["T"]
        )

        total_strength = sum([strength_DT, strength_DD, strength_TT])

        dNdE_TT = strength_TT * nst.dNdE_TT(E_pspec, ion_temperature_kev)
        tt_source = openmc.stats.Tabular(E_pspec * 1e6, dNdE_TT)

        DD_std_dev = np.sqrt(
            DDvar * 1e12
        )  # power 12 as this is in MeV^2 and we need eV
        dd_source = openmc.stats.Normal(mean_value=DDmean * 1e6, std_dev=DD_std_dev)
        # normal could be done with Muir but in this case we have the mean and std dev from NeSST
        # dd_source = openmc.stats.muir(e0=DDmean * 1e6, m_rat=4, kt=ion_temperature)

        DT_std_dev = np.sqrt(
            DTvar * 1e12
        )  # power 12 as this is in MeV^2 and we need eV
        dt_source = openmc.stats.Normal(mean_value=DTmean * 1e6, std_dev=DT_std_dev)
        # normal could be done with Muir but in this case we have the mean and std dev from NeSST
        # dt_source = openmc.stats.muir(e0=DTmean * 1e6, m_rat=5, kt=ion_temperature)

        # todo look into combining distributions openmc.data.combine_distributions()
        return [tt_source, dd_source, dt_source], [
            strength_TT / total_strength,
            strength_DD / total_strength,
            strength_DT / total_strength,
        ]

    elif ["D"] == sorted(set(fuel.keys())):
        strength_DD = 1.0
        dd_source = openmc.stats.muir(e0=DDmean * 1e6, m_rat=4, kt=ion_temperature)
        return [dd_source], [strength_DD]

    elif ["T"] == sorted(set(fuel.keys())):
        strength_TT = 1.0
        dNdE_TT = strength_TT * nst.dNdE_TT(E_pspec, ion_temperature_kev)
        tt_source = openmc.stats.Tabular(E_pspec * 1e6, dNdE_TT)
        return [tt_source], [strength_TT]
