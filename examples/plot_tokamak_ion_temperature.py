import matplotlib.pyplot as plt
import numpy as np

from openmc_plasma_source import tokamak_convert_a_alpha_to_R_Z, tokamak_ion_temperature

sample_size = 20000
minor_radius = 292.258
major_radius = 906

# create a sample of (a, alpha) coordinates
a = np.random.random(sample_size) * minor_radius
alpha = np.random.random(sample_size) * 2 * np.pi

temperatures = tokamak_ion_temperature(
    r=a,
    mode="L",
    pedestal_radius=0.8 * minor_radius,
    ion_temperature_pedestal=6.09,
    ion_temperature_centre=45.9,
    ion_temperature_beta=2,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_separatrix=0.1,
    major_radius=major_radius,
)

RZ = tokamak_convert_a_alpha_to_R_Z(
    a=a,
    alpha=alpha,
    shafranov_factor=0.44789,
    minor_radius=minor_radius,
    major_radius=major_radius,
    triangularity=0.270,
    elongation=1.557,
)

plt.scatter(RZ[0], RZ[1], c=temperatures)
plt.gca().set_aspect("equal")
plt.xlabel("R [cm]")
plt.ylabel("Z [cm]")
plt.colorbar(label="Ion temperature [eV]")

plt.savefig("tokamak_source_ion_temperature.png")
print("written tokamak_source_ion_temperature.png")
