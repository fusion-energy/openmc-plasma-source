from pathlib import Path
import openmc
from openmc_plasma_source import TokamakSource

# just making use of a local cross section xml file, replace with your own cross sections or comment out
openmc.config["cross_sections"] = Path(__file__).parent.resolve() / "cross_sections.xml"


# minimal geometry
sphere_surface = openmc.Sphere(r=1000.0, boundary_type="vacuum")
cell = openmc.Cell(region=-sphere_surface)
geometry = openmc.Geometry([cell])

# create a plasma source
my_plasma = TokamakSource(
    elongation=1.557,
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_pedestal=6.09,
    ion_temperature_separatrix=0.1,
    major_radius=9.06,
    minor_radius=2.92258,
    pedestal_radius=0.8 * 2.92258,
    mode="H",
    shafranov_factor=0.44789,
    triangularity=0.270,
    ion_temperature_beta=6,
)

# Tell OpenMC we're going to use our custom source
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
settings.source = my_plasma.sources


model = openmc.model.Model(materials=None, geometry=geometry, settings=settings)

model.run()
