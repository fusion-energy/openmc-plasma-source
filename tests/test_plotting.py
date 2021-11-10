import matplotlib.pyplot as plt
from openmc_plasma_source import plotting, TokamakSource
import pytest

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
my_plasma.sample_sources()


def test_scatter_tokamak_source():
    plt.figure()
    plotting.scatter_tokamak_source(my_plasma)
    plotting.scatter_tokamak_source(my_plasma, quantity="ion_temperature")
    plotting.scatter_tokamak_source(my_plasma, alpha=0.2)


def test_scatter_tokamak_wrong_quantity():
    plt.figure()
    with pytest.raises(ValueError):
        plotting.scatter_tokamak_source(my_plasma, quantity="coucou")


def test_plot_tokamak_source():
    plt.figure()
    plotting.plot_tokamak_source_3D(my_plasma)
    plotting.plot_tokamak_source_3D(my_plasma, quantity="ion_temperature")
    plotting.plot_tokamak_source_3D(my_plasma, alpha=0.2)
    plotting.plot_tokamak_source_3D(my_plasma, angles=[0, 3.14])


def test_plot_tokamak_wrong_quantity():
    plt.figure()
    with pytest.raises(ValueError):
        plotting.plot_tokamak_source_3D(my_plasma, quantity="coucou")
