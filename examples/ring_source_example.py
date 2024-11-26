import math
from pathlib import Path

import openmc

from openmc_plasma_source import fusion_ring_source

# just making use of a local cross section xml file, replace with your own cross sections or comment out
openmc.config["cross_sections"] = Path(__file__).parent.resolve() / "cross_sections.xml"

# minimal geometry
sphere_surface = openmc.Sphere(r=1000.0, boundary_type="vacuum")
cell = openmc.Cell(region=-sphere_surface)
geometry = openmc.Geometry([cell])

# define the source
my_source = fusion_ring_source(
    radius=700,
    angles=(0.0, 2 * math.pi),  # 360deg source
    temperature=20000.0,
    fuel={"D": 0.5, "T": 0.5},
)

settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
# tell OpenMC we're going to use our custom source
settings.source = my_source

model = openmc.Model(materials=None, geometry=geometry, settings=settings)

model.run()


# optionally if you would like to plot the location of particles then another package can be used
# https://github.com/fusion-energy/openmc_source_plotter

from openmc_source_plotter import plot_source_position

plot = plot_source_position(
    this=settings,
    n_samples=2000,
)

plot.show()
