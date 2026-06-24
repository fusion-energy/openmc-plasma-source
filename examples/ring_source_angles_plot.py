"""Plot fusion_ring_source positions in 3D for a few start_angle / rotation_angle
combinations.

This samples the source's CylindricalIndependent space distribution directly and
builds a 3D plotly figure, so it needs no nuclear data or openmc.lib runtime. For
sampling via OpenMC's own machinery instead, see
openmc_source_plotter.plot_source_position.
"""

import numpy as np
import plotly.graph_objects as go

from openmc_plasma_source import fusion_ring_source

N_SAMPLES = 2000

# A few (start_angle, rotation_angle) combinations, each placed at its own
# z height so the toroidal sectors are easy to tell apart in 3D.
cases = [
    {
        "start_angle": 0.0,
        "rotation_angle": 2 * np.pi,
        "z": 0,
        "label": "full ring (0 -> 2pi)",
    },
    {
        "start_angle": 0.0,
        "rotation_angle": np.pi / 2,
        "z": 200,
        "label": "start=0, rot=pi/2",
    },
    {
        "start_angle": np.pi,
        "rotation_angle": np.pi / 2,
        "z": 400,
        "label": "start=pi, rot=pi/2",
    },
    {
        "start_angle": np.pi / 2,
        "rotation_angle": np.pi,
        "z": 600,
        "label": "start=pi/2, rot=pi",
    },
    {
        "start_angle": 0.0,
        "rotation_angle": -np.pi / 2,
        "z": 800,
        "label": "start=0, rot=-pi/2",
    },
]

figure = go.Figure()
for i, case in enumerate(cases):
    source = fusion_ring_source(
        radius=700,
        start_angle=case["start_angle"],
        rotation_angle=case["rotation_angle"],
        z_placement=case["z"],
    )[0]

    space = source.space
    r = space.r.sample(N_SAMPLES, seed=i + 1)
    phi = space.phi.sample(N_SAMPLES, seed=i + 1)
    z = space.z.sample(N_SAMPLES, seed=i + 1)

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={"size": 2},
            name=case["label"],
        )
    )

figure.update_layout(
    title="fusion_ring_source: start_angle / rotation_angle variations",
    legend_title="case",
    scene={
        "xaxis_title": "x (cm)",
        "yaxis_title": "y (cm)",
        "zaxis_title": "z (cm)",
        "aspectmode": "data",
    },
)

figure.write_html("ring_source_angles.html")
print("wrote ring_source_angles.html")

# Static preview with matplotlib (avoids needing a browser for plotly image export).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(projection="3d")
for trace in figure.data:
    ax.scatter(trace.x, trace.y, trace.z, s=2, label=trace.name)
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_zlabel("z (cm)")
ax.set_title("fusion_ring_source: start_angle / rotation_angle variations")
ax.legend(loc="upper left", fontsize=8)
ax.view_init(elev=35, azim=-60)
fig.savefig("ring_source_angles.png", dpi=120, bbox_inches="tight")
print("wrote ring_source_angles.png")
