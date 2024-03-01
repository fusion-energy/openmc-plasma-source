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

    if ["D"] == sorted(set(fuel.keys())):
        max_energy_mev = 5
    elif ["T"] == sorted(set(fuel.keys())):
        max_energy_mev = 12
    elif ["D", "T"] == sorted(set(fuel.keys())):
        max_energy_mev = 20

    print(max_energy_mev, "MeV")

    # 1.0 neutron yield, all reactions scaled by this value
    num_of_vals = 50
    # single grid for DT, DD and TT grid
    E_pspec = np.linspace(0, 12e6, num_of_vals)  # accepts MeV units

    dNdE_DT_DD_TT = np.zeros(num_of_vals)
    if ["D", "T"] == sorted(set(fuel.keys())):
        # DTmean, DTvar = nst.DTprimspecmoments(ion_temperature_kev)
        # DDmean, DDvar = nst.DDprimspecmoments(ion_temperature_kev)

        Y_DT = 1.0
        Y_DD = nst.yield_from_dt_yield_ratio(
            "dd", Y_DT, ion_temperature_kev, fuel["D"], fuel["T"]
        )
        Y_TT = nst.yield_from_dt_yield_ratio(
            "tt", Y_DT, ion_temperature_kev, fuel["D"], fuel["T"]
        )

        # dNdE_DT = Y_DT * nst.Qb(E_pspec, DTmean, DTvar)  # Brysk shape i.e. Gaussian
        # dNdE_DD = Y_DD * nst.Qb(E_pspec, DDmean, DDvar)  # Brysk shape i.e. Gaussian
        dNdE_TT = Y_TT * nst.dNdE_TT(E_pspec, ion_temperature_kev)
        # dNdE_DT_DD_TT = dNdE_DT + dNdE_DD + dNdE_TT
        dNdE_DT_DD_TT = dNdE_TT
        print("dt")
        tt_source = openmc.stats.Discrete(E_pspec * 1e6, dNdE_DT_DD_TT)
        dd_source = openmc.stats.muir(e0=2.5e6, m_rat=4, kt=ion_temperature)
        dt_source = openmc.stats.muir(e0=14.06e6, m_rat=4, kt=ion_temperature)
        return [tt_source, dd_source, dt_source]
    elif ["D"] == sorted(set(fuel.keys())):
        DTmean, DTvar = nst.DTprimspecmoments(ion_temperature_kev)
        DDmean, DDvar = nst.DDprimspecmoments(ion_temperature_kev)

        print("d")
        Y_DD = 1.0

        dNdE_DD = Y_DD * nst.Qb(E_pspec, DDmean, DDvar)  # Brysk shape i.e. Gaussian
        return openmc.stats.Discrete(E_pspec * 1e6, dNdE_DD)

    elif ["T"] == sorted(set(fuel.keys())):
        Y_TT = 1.0

        print("t")
        dNdE_TT = Y_TT * nst.dNdE_TT(E_pspec, ion_temperature_kev)
        return openmc.stats.Discrete(E_pspec * 1e6, dNdE_TT)
