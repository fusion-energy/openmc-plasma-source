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
    fuel={"D": 0.5, "T": 0.5},
)

# Tell OpenMC we're going to use our custom source
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
settings.source = my_source

model = openmc.model.Model(materials=None, geometry=geometry, settings=settings)

model.run()


# optionally if you would like to plot the energy of particles then another package can be used
# https://github.com/fusion-energy/openmc_source_plotter

# from openmc_source_plotter import plot_source_energy

# plot = plot_source_energy(
#     this=settings,
#     n_samples=1000000, # increase this value for a smoother plot
#     energy_bins=np.linspace(0, 16e6, 1000),
#     yaxis_type="log",
# )
# plot.show()
