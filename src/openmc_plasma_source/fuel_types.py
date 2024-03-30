import NeSST as nst
import numpy as np
import openmc


def neutron_energy_mean(ion_temperature: float, reaction: str) -> float:
    """Calculates the mean energy of the neutron emitted during DD or DT
    fusion accounting for temperature of the incident ions. Based on Ballabio
    fits, see Table III of L. Ballabio et al 1998 Nucl. Fusion 38 1723

    Args:
        ion_temperature (float): the temperature of the ions in eV
        reaction (str): the two isotope that fuse, can be either 'DD' or 'DT'

    Raises:
        ValueError: if the reaction is not 'DD' or 'DT' then a ValueError is raised

    Returns:
        float: the mean neutron energy in eV
    """

    # values from Ballabio paper
    if reaction == "DD":
        a_1 = 4.69515
        a_2 = -0.040729
        a_3 = 0.47
        a_4 = 0.81844
        mean = 2.4495e6  # in units of eV
    elif reaction == "DT":
        a_1 = 5.30509
        a_2 = 2.4736e-3
        a_3 = 1.84
        a_4 = 1.3818
        mean = 14.021e6  # in units of eV
    else:
        raise ValueError(f'reaction must be either "DD" or "DT" not {reaction}')

    ion_temperature_kev = ion_temperature / 1e3  # Ballabio equation accepts KeV units
    mean_delta = (
        a_1
        * ion_temperature_kev ** (2.0 / 3.0)
        / (1.0 + a_2 * ion_temperature_kev**a_3)
        + a_4 * ion_temperature_kev
    )
    mean_delta *= 1e3  # converting back to eV
    return mean + mean_delta


def neutron_energy_std_dev(ion_temperature: float, reaction: str) -> float:
    """Calculates the standard deviation of the neutron energy emitted during DD
     or DT fusion accounting for temperature of the incident ions. Based on
    Ballabio fits, see Table III of L. Ballabio et al 1998 Nucl. Fusion 38 1723

    Args:
        ion_temperature (float): the temperature of the ions in eV
        reaction (str): the two isotope that fuse, can be either 'DD' or 'DT'

    Raises:
        ValueError: if the reaction is not 'DD' or 'DT' then a ValueError is raised

    Returns:
        float: the mean neutron energy in eV
    """

    # values from Ballabio paper
    if reaction == "DD":
        w_0 = 82.542
        a_1 = 1.7013e-3
        a_2 = 0.16888
        a_3 = 0.49
        a_4 = 7.9460e-4
    elif reaction == "DT":
        w_0 = 177.259
        a_1 = 5.1068e-4
        a_2 = 7.6223e-3
        a_3 = 1.78
        a_4 = 8.7691e-5
    else:
        raise ValueError(f'reaction must be either "DD" or "DT" not {reaction}')

    ion_temperature_kev = ion_temperature / 1e3  # Ballabio equation accepts KeV units
    delta = (
        a_1
        * ion_temperature_kev ** (2.0 / 3.0)
        / (1.0 + a_2 * ion_temperature_kev**a_3)
        + a_4 * ion_temperature_kev
    )

    # 2.3548200450309493 on the line below comes from equation 2* math.sqrt(math.log(2)*2)
    variance = ((w_0 * (1 + delta)) ** 2 * ion_temperature_kev) / 2.3548200450309493**2
    variance *= 1e6  # converting keV^2 back to eV^2
    std_dev = np.sqrt(variance)
    return std_dev


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

    DDmean = neutron_energy_mean(ion_temperature=ion_temperature, reaction="DD")
    DTmean = neutron_energy_mean(ion_temperature=ion_temperature, reaction="DT")
    DD_std_dev = neutron_energy_std_dev(ion_temperature=ion_temperature, reaction="DD")
    DT_std_dev = neutron_energy_std_dev(ion_temperature=ion_temperature, reaction="DT")

    if ["D", "T"] == sorted(set(fuel.keys())):
        strength_DT = 1.0
        strength_DD = nst.yield_from_dt_yield_ratio(
            "dd", strength_DT, ion_temperature, fuel["D"], fuel["T"]
        )
        strength_TT = nst.yield_from_dt_yield_ratio(
            "tt", strength_DT, ion_temperature, fuel["D"], fuel["T"]
        )

        total_strength = sum([strength_TT, strength_DD, strength_DT])

        dNdE_TT = strength_TT * nst.dNdE_TT(E_pspec, ion_temperature)
        tt_source = openmc.stats.Tabular(E_pspec * 1e6, dNdE_TT)

        dd_source = openmc.stats.Normal(mean_value=DDmean, std_dev=DD_std_dev)
        # normal could be done with Muir but in this case we have the mean and std dev from NeSST
        # dd_source = openmc.stats.muir(e0=DDmean * 1e6, m_rat=4, kt=ion_temperature)

        dt_source = openmc.stats.Normal(mean_value=DTmean, std_dev=DT_std_dev)
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
        dNdE_TT = strength_TT * nst.dNdE_TT(E_pspec, ion_temperature)
        tt_source = openmc.stats.Tabular(E_pspec * 1e6, dNdE_TT)
        return [tt_source], [strength_TT]
