import matplotlib.pyplot as plt
from openmc_plasma_source import fusion_point_source

import numpy as np

tritium_factions = np.linspace(0, 1, 3)
tritium_factions = np.concatenate([tritium_factions, [0.99]])
tritium_factions = np.sort(tritium_factions)

for tritium_fraction in tritium_factions:
    fuel = {"D": 1 - tritium_fraction, "T": tritium_fraction}

    if fuel["D"] == 0:
        fuel = {"T": 1}
    if fuel["T"] == 0:
        fuel = {"D": 1}
    my_source = fusion_point_source(
        coordinate=(0, 0, 0), temperature=20000.0, fuel=fuel
    )

    # Plot the source energy distribution
    energies = my_source[0].energy.sample(n_samples=int(5e6))

    data, bins, _ = plt.hist(
        energies / 1e6,
        bins=1000,
        histtype="step",
        label=f"{1-tritium_fraction:.0%} D - {tritium_fraction:.0%} T",
        density=True,
        # alpha=0.5,
    )

plt.xlabel("Energy (MeV)")
plt.ylabel("Neturon energy distribution")
plt.yscale("log")
plt.legend()
plt.ylim(bottom=1e-3)
plt.savefig("energy_spectra.svg")
plt.show()
