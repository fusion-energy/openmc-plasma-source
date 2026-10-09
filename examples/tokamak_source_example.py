from pathlib import Path

import openmc

from openmc_plasma_source import tokamak_source

# just making use of a local cross section xml file, replace with your own cross sections or comment out
openmc.config["cross_sections"] = Path(__file__).parent.resolve() / "cross_sections.xml"


# minimal geometry
sphere_surface = openmc.Sphere(r=100000.0, boundary_type="vacuum")
cell = openmc.Cell(region=-sphere_surface)
geometry = openmc.Geometry([cell])

# create a plasma source
make_source = tokamak_source(
    elongation=1.557,
    ion_density_centre=1.09e20,
    ion_density_pedestal=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9e3,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6,
    major_radius=906,
    minor_radius=292.258,
    pedestal_radius=0.8 * 292.258,
    mode="H",
    shafranov_factor=0.44789,
    triangularity=0.270,
    fuel={"D": 0.5, "T": 0.5},
)
my_source = make_source.source

# Tell OpenMC we're going to use our custom source
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
settings.source = [my_source]


model = openmc.model.Model(materials=None, geometry=geometry, settings=settings)

model.run()


# optionally if you would like to plot the direction of particles then another package can be used
# https://github.com/fusion-energy/openmc_source_plotter

from openmc_source_plotter import plot_source_position

plot = plot_source_position(this=settings, n_samples=200)

# equal scaling on all three axes, otherwise plotly stretches each axis to the
# same length and the plasma looks much taller than it is
plot.update_layout(scene={"aspectmode": "data"})

plot.show()
