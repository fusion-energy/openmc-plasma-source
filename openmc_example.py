import openmc
from main import Plasma


# create a plasma source
my_plasma = Plasma(
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
    ion_temperature_beta=6
    )
my_plasma.sample_sources()

my_sources = []
# create a ring source for each sample in the plasma source
for i in range(len(my_plasma.strengths)):
    my_source = openmc.Source()

    # extract the RZ values accordingly
    radius = openmc.stats.Discrete([my_plasma.RZ[0][i]], [1])
    z_values = openmc.stats.Discrete([my_plasma.RZ[1][i]], [1])

    # full rotation
    angle = openmc.stats.Uniform(a=0., b=2*3.14159265359)
    # create a ring source
    my_source.space = openmc.stats.CylindricalIndependent(r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0))

    my_source.angle = openmc.stats.Isotropic()
    my_source.energy = openmc.stats.Muir(e0=14080000.0, m_rat=5.0, kt=my_plasma.temperatures[i])

    # the strength of the source (its probability) is given by my_plasma.strengths
    my_source.strength = my_plasma.strengths[i]

    # append to the list of sources
    my_sources.append(my_source)


# Create a single material
iron = openmc.Material()
iron.set_density("g/cm3", 5.0)
iron.add_element("Fe", 1.0)
mats = openmc.Materials([iron])

# Create a 5 cm x 5 cm box filled with iron
cells = []
inner_box1 = openmc.ZCylinder(r=600.0)
inner_box2 = openmc.ZCylinder(r=1400.0)
outer_box = openmc.model.rectangular_prism(4000.0, 4000.0, boundary_type="vacuum")
cells += [openmc.Cell(fill=iron, region=-inner_box1)]
cells += [openmc.Cell(fill=None, region=+inner_box1 & -inner_box2)]
cells += [openmc.Cell(fill=iron, region=+inner_box2 & outer_box)]
geometry = openmc.Geometry(cells)

# Tell OpenMC we're going to use our custom source
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 1000
settings.source = my_sources

# Finally, define a mesh tally so that we can see the resulting flux
mesh = openmc.RegularMesh()
mesh.lower_left = (-2000.0, -2000.0)
mesh.upper_right = (2000.0, 2000.0)
mesh.dimension = (50, 50)

tally = openmc.Tally()
tally.filters = [openmc.MeshFilter(mesh)]
tally.scores = ["flux"]
tallies = openmc.Tallies([tally])

model = openmc.model.Model(
    materials=mats, geometry=geometry, settings=settings, tallies=tallies
)

model.run()
