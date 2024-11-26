import matplotlib.pyplot as plt
import numpy as np

from openmc_plasma_source import (
    tokamak_convert_a_alpha_to_R_Z,
    tokamak_ion_density,
    tokamak_ion_temperature,
    tokamak_neutron_source_density,
)

sample_size = 20000
minor_radius = 292.258
major_radius = 906
mode = "L"
ion_density_centre = 45.9

# create a sample of (a, alpha) coordinates
a = np.random.random(sample_size) * minor_radius
alpha = np.random.random(sample_size) * 2 * np.pi

temperatures = tokamak_ion_temperature(
    r=a,
    mode=mode,
    pedestal_radius=0.8 * minor_radius,
    ion_temperature_pedestal=6.09,
    ion_temperature_centre=ion_density_centre,
    ion_temperature_beta=2,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_separatrix=0.1,
    major_radius=major_radius,
)

densities = tokamak_ion_density(
    mode=mode,
    ion_density_centre=ion_density_centre,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    major_radius=major_radius,
    pedestal_radius=0.8 * minor_radius,
    ion_density_separatrix=3e19,
    r=a,
)

neutron_source_density = tokamak_neutron_source_density(densities, temperatures, "DD")

strengths = neutron_source_density / sum(neutron_source_density)

RZ = tokamak_convert_a_alpha_to_R_Z(
    a=a,
    alpha=alpha,
    shafranov_factor=0.44789,
    minor_radius=minor_radius,
    major_radius=major_radius,
    triangularity=0.270,
    elongation=1.557,
)

plt.scatter(RZ[0], RZ[1], c=strengths)
plt.gca().set_aspect("equal")
plt.xlabel("R [cm]")
plt.ylabel("Z [cm]")
plt.colorbar(label="neutron emission strength")

plt.savefig("tokamak_source_neutron_emission_strength.png")
print("written tokamak_source_neutron_emission_strength.png")
