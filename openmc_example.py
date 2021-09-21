import openmc
import openmc_dagmc_wrapper
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
    my_source.energy = openmc.stats.Muir(e0=14080000.0, m_rat=5.0, kt=20000.0)

    # the strength of the source (its probability) is given by my_plasma.strengths
    my_source.strength = my_plasma.strengths[i]

    # append to the list of sources
    my_sources.append(my_source)


neutronics_model = openmc_dagmc_wrapper.NeutronicsModel(
    h5m_filename='stage_2_output/dagmc.h5m',
    source=my_sources,  # openmc.Settings().source can accept a list of sources
    materials={"mat1": "eurofer"},
    cell_tallies=["flux"]
)

neutronics_model.simulate(
    simulation_batches=5,
    simulation_particles_per_batch=1e4,
)

neutronics_model.process_results(
    cell_tally_results_filename='results.json'
)
