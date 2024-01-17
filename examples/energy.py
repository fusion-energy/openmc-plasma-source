import matplotlib.pyplot as plt
import NeSST as nst
import numpy as np
import openmc
import openmc_source_plotter  # extends openmc.Source with plotting functions

Ion_temperature = 2.0  # keV

# Atomic fraction of D and T in scattering medium and source
nst.frac_D_default = 0.5
nst.frac_T_default = 0.5

DTmean, DTvar = nst.DTprimspecmoments(Ion_temperature)
DDmean, DDvar = nst.DDprimspecmoments(Ion_temperature)

Y_DT = 1.0  # DT neutron yield, all reactions scaled by this value
Y_DD = nst.yield_from_dt_yield_ratio("dd", Y_DT, Ion_temperature)
Y_TT = nst.yield_from_dt_yield_ratio("tt", Y_DT, Ion_temperature)

# single grid for DT, DD and TT grid
E_pspec = np.linspace(0, 20, 500)

dNdE_DT = Y_DT * nst.Qb(E_pspec, DTmean, DTvar)  # Brysk shape i.e. Gaussian
dNdE_DD = Y_DD * nst.Qb(E_pspec, DDmean, DDvar)  # Brysk shape i.e. Gaussian
dNdE_TT = Y_TT * nst.dNdE_TT(E_pspec, Ion_temperature)

dNdE_DT_DD_TT = dNdE_DT + dNdE_DD + dNdE_TT

my_source = openmc.IndependentSource()
my_source.space = openmc.stats.Point((0, 0, 0))
my_source.angle = openmc.stats.Isotropic()
my_source.energy = openmc.stats.Discrete(E_pspec * 1e6, dNdE_DT_DD_TT)
plot = my_source.plot_source_energy(n_samples=200000)

plot.update_yaxes(type="log")

plot.show()
