import openmc
from openmc_plasma_source import fusion_point_source
from pathlib import Path
import openmc_source_plotter

# just making use of a local cross section xml file, replace with your own cross sections or comment out
openmc.config["cross_sections"] = Path(__file__).parent.resolve() / "cross_sections.xml"

# minimal geometry
sphere_surface = openmc.Sphere(r=1000.0, boundary_type="vacuum")
cell = openmc.Cell(region=-sphere_surface)
geometry = openmc.Geometry([cell])

# define the source
my_source = fusion_point_source(
    # coordinate=(0, 0, 0), temperature=20000.0, fuel={"D": 0.01, "T": 0.99}
    coordinate=(0, 0, 0),
    temperature=20000.0,
    fuel={"D": 0.5, "T": 0.5},
    # coordinate=(0, 0, 0), temperature=20000.0, fuel={"D": 0.99, "T": 0.01}
    # coordinate=(0, 0, 0), temperature=20000.0, fuel={"D": 1.}
    # coordinate=(0, 0, 0), temperature=20000.0, fuel={"T": 1.}
)

# Tell OpenMC we're going to use our custom source
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
settings.source = my_source

import numpy as np

plot = settings.plot_source_energy(
    n_samples=100000,
    energy_bins=np.linspace(0, 16e6, 1000),
    yaxis_type="log",
)
plot.show()

model = openmc.model.Model(materials=None, geometry=geometry, settings=settings)

model.run()
