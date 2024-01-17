import openmc
from openmc_plasma_source import FusionPointSource

# minimal geometry
sphere_surface = openmc.Sphere(r=1000.0, boundary_type="vacuum")
cell = openmc.Cell(region=-sphere_surface)
geometry = openmc.Geometry([cell])

# define the source
my_source = FusionPointSource(coordinate=(0, 0, 0), temperature=20000.0, fuel="DT")

# Tell OpenMC we're going to use our custom source
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
settings.source = my_source


model = openmc.model.Model(materials=None, geometry=geometry, settings=settings)

model.run()
