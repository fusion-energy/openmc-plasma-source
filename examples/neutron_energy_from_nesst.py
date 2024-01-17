import numpy as np
import matplotlib.pyplot as plt
import NeSST as nst
import openmc

# Atomic fraction of D and T in scattering medium and source
nst.frac_D_default = 0.5
nst.frac_T_default = 0.5

Tion = 20.0 # keV
Y_DT = 1.0 # DT neutron yield, all reactions scaled by this value

DTmean,DTvar = nst.DTprimspecmoments(Tion)
DDmean,DDvar = nst.DDprimspecmoments(Tion)

Y_DD = nst.yield_from_dt_yield_ratio('dd',Y_DT,Tion)
Y_TT = nst.yield_from_dt_yield_ratio('tt',Y_DT,Tion)

# DD and TT grid
E_pspec = np.linspace(0,10.0,300)
# DT grid
E_DTspec = np.linspace(12.5,15.5,200)
E_pspec = np.append(E_pspec,E_DTspec)

dNdE_DT = Y_DT*nst.Qb(E_pspec,DTmean,DTvar) # Brysk shape i.e. Gaussian
dNdE_DD = Y_DD*nst.Qb(E_pspec,DDmean,DDvar) # Brysk shape i.e. Gaussian
dNdE_TT = Y_TT*nst.dNdE_TT(E_pspec,Tion)

plt.figure(dpi=200)

plt.semilogy(E_pspec,dNdE_DT)
plt.semilogy(E_pspec,dNdE_DD)
plt.semilogy(E_pspec,dNdE_TT)

plt.xlim(E_pspec[0],E_pspec[-1])
plt.ylim(1e-4,1e1)
plt.xlabel("Energy (MeV)")
plt.ylabel("dN/dE (1/MeV)")
plt.show()

my_source_4 = openmc.IndependentSource()
my_source_4.space = openmc.stats.Point((0, 0, 0))
my_source_4.angle = openmc.stats.Isotropic()
my_source_4.energy = openmc.stats.Discrete([14e6], [1])