"""Plot tokamak (plasma) sources in 3D, including a seam-crossing sector.

tokamak_source returns an openmc.MeshSource backed by a CylindricalMesh: each
voxel is an IndependentSource whose strength is proportional to the local
neutron emission. The poloidal (R, Z) emission profile is identical in every
toroidal bin, so we build the source cheaply with a single phi bin to get the
(R, Z) strength map, then expand it into a 3D toroidal point cloud using the
real toroidal grid from _toroidal_phi_grid. This faithfully reproduces the
sector geometry, including the zero-strength "dead" bin of a sector that
crosses the 0 / 2*pi seam.

This samples the strength map directly and needs no nuclear data or openmc.lib
runtime. For sampling via OpenMC's own machinery instead, see
openmc_source_plotter.plot_source_position.
"""

import numpy as np
import plotly.graph_objects as go

from openmc_plasma_source import tokamak_source
from openmc_plasma_source.tokamak_source import _toroidal_phi_grid

N_SAMPLES = 30000
N_PHI = 48
RNG = np.random.default_rng(1)

# ITER-like plasma parameters (same as examples/tokamak_source_example.py)
PLASMA = dict(
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

# The two source cases to plot
cases = [
    {
        "start_angle": 0.0,
        "rotation_angle": 1.5 * np.pi,
        "label": "0 -> 270deg (cutaway)",
    },
    {
        "start_angle": 7 * np.pi / 4,
        "rotation_angle": np.pi / 2,
        "label": "315 -> 45deg (crosses seam)",
    },
]


def sample_points(start_angle, rotation_angle):
    """Sample neutron birth positions for a tokamak sector using the real mesh."""
    # Cheap single-phi-bin build to get the poloidal (R, Z) strength map.
    source = tokamak_source(
        **PLASMA,
        start_angle=0.0,
        rotation_angle=2 * np.pi,
        mesh_resolution=(100, 1, 100),
    )
    n_r = len(source.mesh.r_grid) - 1
    n_z = len(source.mesh.z_grid) - 1
    strength = np.array([s.strength for s in source.sources.ravel()]).reshape(
        n_r, 1, n_z
    )[:, 0, :]

    r_edges = np.asarray(source.mesh.r_grid)
    z_edges = np.asarray(source.mesh.z_grid)
    R_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    Z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dr = r_edges[1] - r_edges[0]
    dz = z_edges[1] - z_edges[0]

    # Real toroidal grid for this sector (handles seam crossing + dead bin).
    phi_grid, phi_fraction = _toroidal_phi_grid(start_angle, rotation_angle, N_PHI)

    # Choose poloidal cells in proportion to emission strength.
    i_idx, k_idx = np.nonzero(strength)
    p_pol = strength[i_idx, k_idx]
    p_pol = p_pol / p_pol.sum()
    pol_choice = RNG.choice(len(i_idx), size=N_SAMPLES, p=p_pol)
    ii = i_idx[pol_choice]
    kk = k_idx[pol_choice]

    # Choose phi bins in proportion to their (active) strength fraction.
    active = phi_fraction > 0
    bin_ids = np.nonzero(active)[0]
    p_phi = phi_fraction[active] / phi_fraction[active].sum()
    phi_choice = RNG.choice(bin_ids, size=N_SAMPLES, p=p_phi)
    phi = RNG.uniform(phi_grid[phi_choice], phi_grid[phi_choice + 1])

    R = R_centers[ii] + (RNG.random(N_SAMPLES) - 0.5) * dr
    Z = Z_centers[kk] + (RNG.random(N_SAMPLES) - 0.5) * dz
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    return x, y, Z, strength[ii, kk]


figure = go.Figure()
for case in cases:
    x, y, z, s = sample_points(case["start_angle"], case["rotation_angle"])
    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={"size": 1.5, "color": s, "colorscale": "Inferno", "opacity": 0.5},
            name=case["label"],
        )
    )
figure.update_layout(
    title="tokamak_source: toroidal sectors (colored by neutron emission strength)",
    legend_title="case",
    scene={
        "xaxis_title": "x (cm)",
        "yaxis_title": "y (cm)",
        "zaxis_title": "z (cm)",
        "aspectmode": "data",
    },
)
figure.write_html("tokamak_source.html")
print("wrote tokamak_source.html")

# Static preview with matplotlib (no browser needed).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(13, 5))
for col, case in enumerate(cases):
    x, y, z, s = sample_points(case["start_angle"], case["rotation_angle"])
    ax = fig.add_subplot(1, 2, col + 1, projection="3d")
    ax.scatter(x, y, z, c=s, cmap="inferno", s=1, alpha=0.4)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_title(case["label"])
    ax.view_init(elev=35, azim=-50)
fig.suptitle("tokamak_source neutron emission")
fig.savefig("tokamak_source.png", dpi=120, bbox_inches="tight")
print("wrote tokamak_source.png")
