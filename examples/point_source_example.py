from pathlib import Path

import numpy as np
import openmc

from openmc_plasma_source import fusion_point_source

# just making use of a local cross section xml file, replace with your own cross sections or comment out
openmc.config["cross_sections"] = Path(__file__).parent.resolve() / "cross_sections.xml"

# minimal geometry
sphere_surface = openmc.Sphere(r=1000.0, boundary_type="vacuum")
cell = openmc.Cell(region=-sphere_surface)
geometry = openmc.Geometry([cell])

# define the source
my_source = fusion_point_source(
    coordinate=(0, 0, 0),
    temperature=20000.0,
    fuel={
        "D": 0.09,
        "T": 0.91,
    },  # note this is mainly tritium fuel so that TT reactions are more likely
)


# Tell OpenMC we're going to use our custom source
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
settings.source = my_source

model = openmc.model.Model(materials=None, geometry=geometry, settings=settings)

model.run()

# Plot the source energy distribution
energies = my_source[0].energy.sample(n_samples=10000)
import matplotlib.pyplot as plt

plt.hist(energies, bins=1000)
plt.xlim(14e6 - 2e6, 14e6 + 2e6)
plt.show()
