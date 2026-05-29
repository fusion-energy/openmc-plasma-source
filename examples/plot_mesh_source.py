import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from openmc_plasma_source import tokamak_source, tokamak_convert_a_alpha_to_R_Z

mesh_source = tokamak_source(
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
    mesh_resolution=(100, 1, 100),
    grid_density=500,
)

# Extract mesh grid edges and strengths
r_grid = np.asarray(mesh_source.mesh.r_grid)
z_grid = np.asarray(mesh_source.mesh.z_grid)

# Build 2D strength array — MeshSource flattens to 1D (n_r * n_phi * n_z)
sources = mesh_source.sources
n_r = len(r_grid) - 1
n_z = len(z_grid) - 1
n_phi = sources.size // (n_r * n_z)
sources_3d = sources.reshape(n_r, n_phi, n_z)
strengths = np.zeros((n_r, n_z))
for i in range(n_r):
    for k in range(n_z):
        strengths[i, k] = sources_3d[i, 0, k].strength

# Mask zero-strength voxels for log scale
strengths_masked = np.ma.masked_where(strengths <= 0, strengths)

# LCFS boundary
alpha_lcfs = np.linspace(0, 2 * np.pi, 500)
a_lcfs = np.full_like(alpha_lcfs, 292.258)
R_lcfs, Z_lcfs = tokamak_convert_a_alpha_to_R_Z(
    a=a_lcfs,
    alpha=alpha_lcfs,
    shafranov_factor=0.44789,
    minor_radius=292.258,
    major_radius=906,
    triangularity=0.270,
    elongation=1.557,
)

fig, ax = plt.subplots(figsize=(8, 10))
R, Z = np.meshgrid(r_grid, z_grid, indexing="ij")
pcm = ax.pcolormesh(
    R,
    Z,
    strengths_masked,
    shading="flat",
    cmap="plasma",
    norm=mcolors.LogNorm(vmin=strengths_masked.min(), vmax=strengths_masked.max()),
)
ax.plot(R_lcfs, Z_lcfs, "w--", linewidth=1.5, label="LCFS")
ax.set_aspect("equal")
ax.set_xlabel("R [cm]")
ax.set_ylabel("Z [cm]")
ax.set_title("Tokamak MeshSource — Neutron Emission Strength")
ax.legend()
fig.colorbar(pcm, ax=ax, label="source strength per voxel")

fig.savefig("tokamak_mesh_source.png", dpi=150, bbox_inches="tight")
print("Saved tokamak_mesh_source.png")
