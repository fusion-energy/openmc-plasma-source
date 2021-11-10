import openmc
from openmc_plasma_source import FusionRingSource

my_source = FusionRingSource(
    radius=700,
    angles=(0.0, 6.28318530718),  # 360deg source
    temperature=20000.0,
)

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
settings.source = my_source

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
