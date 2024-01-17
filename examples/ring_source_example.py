import openmc
from openmc_plasma_source import FusionRingSource
import math
from pathlib import Path

# just making use of a local cross section xml file, replace with your own cross sections or comment out
openmc.config["cross_sections"] = Path(__file__).parent.resolve() / "cross_sections.xml"

# minimal geometry
sphere_surface = openmc.Sphere(r=1000.0, boundary_type="vacuum")
cell = openmc.Cell(region=-sphere_surface)
geometry = openmc.Geometry([cell])

# define the source
my_source = FusionRingSource(
    radius=700,
    angles=(0.0, 2 * math.pi),  # 360deg source
    temperature=20000.0,
)

settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
# tell OpenMC we're going to use our custom source
settings.source = my_source

model = openmc.model.Model(materials=None, geometry=geometry, settings=settings)

model.run()
