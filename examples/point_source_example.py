import openmc
from openmc_plasma_source import FusionPointSource
from pathlib import Path
import openmc_source_plotter

# just making use of a local cross section xml file, replace with your own cross sections or comment out
openmc.config["cross_sections"] = Path(__file__).parent.resolve() / "cross_sections.xml"

# minimal geometry
sphere_surface = openmc.Sphere(r=1000.0, boundary_type="vacuum")
cell = openmc.Cell(region=-sphere_surface)
geometry = openmc.Geometry([cell])

# define the source
plot= my_source = FusionPointSource(
    coordinate=(0, 0, 0), temperature=20000.0, fuel={"T": 1.}
    # coordinate=(0, 0, 0), temperature=20000.0, fuel={"D": 0.5, "T": 1.}
).plot_source_energy(n_samples=10000)
plot.show()
# Tell OpenMC we're going to use our custom source
# settings = openmc.Settings()
# settings.run_mode = "fixed source"
# settings.batches = 10
# settings.particles = 1000
# settings.source = my_source


# model = openmc.model.Model(materials=None, geometry=geometry, settings=settings)

# model.run()
