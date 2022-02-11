import matplotlib.pyplot as plt
import numpy as np
from openmc_plasma_source import (
    TokamakSource,
    plotting as ops_plt,
)
import pytest


@pytest.fixture
def tokamak_source():
    return TokamakSource(
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
        sample_size=500,
    )


def test_scatter_tokamak_source_defaults(tokamak_source):
    """Ensure plotting is successful without providing additional args"""
    plt.figure()
    assert not plt.gca().collections  # Check current ax is empty
    ops_plt.scatter_tokamak_source(tokamak_source)
    assert plt.gca().collections  # Check current ax is not empty
    # Save for viewing, clean up
    plt.xlabel("R")
    plt.ylabel("Z", rotation=0)
    plt.savefig("tests/test_scatter_tokamak_source_defaults.png")
    plt.close()


def test_scatter_tokamak_source_with_ax(tokamak_source):
    """Ensure plotting is successful for user-provided ax"""
    fig = plt.figure()
    ax = fig.gca()
    assert not ax.collections  # Check ax is empty
    ops_plt.scatter_tokamak_source(tokamak_source, ax=ax)
    assert ax.collections  # Check ax is not empty
    # Save for viewing, clean up
    ax.set_xlabel("R")
    ax.set_ylabel("Z", rotation=0)
    fig.savefig("tests/test_scatter_tokamak_source_with_ax.png")
    plt.close(fig)


def test_scatter_tokamak_source_with_subplots(tokamak_source):
    """Ensure plotting is successful for multiple user-provided ax"""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Plot on the first axes
    assert not ax1.collections  # Check ax is empty
    ops_plt.scatter_tokamak_source(tokamak_source, ax=ax1)
    assert ax1.collections  # Check ax is not empty
    # Generate new data
    tokamak_source.sample_sources()
    tokamak_source.sources = tokamak_source.make_openmc_sources()
    # Plot on the other axes
    assert not ax2.collections  # Check ax is empty
    ops_plt.scatter_tokamak_source(tokamak_source, ax=ax2)
    assert ax2.collections  # Check ax is not empty
    # Save for viewing, clean up
    ax1.set_xlabel("R")
    ax1.set_ylabel("Z", rotation=0)
    ax2.set_xlabel("R")
    fig.savefig("tests/test_scatter_tokamak_source_subplots.png")
    plt.close(fig)


@pytest.mark.parametrize(
    "quantity", ["ion_temperature", "neutron_source_density", "strength"]
)
def test_scatter_tokamak_source_quantities(tokamak_source, quantity):
    """Plot with colours set by 'quantity'"""
    fig = plt.figure()
    ax = fig.gca()
    assert not ax.collections  # Check ax is empty
    ops_plt.scatter_tokamak_source(tokamak_source, ax=ax, quantity=quantity)
    assert ax.collections  # Check ax is not empty
    # Save for viewing, clean up
    ax.set_xlabel("R")
    ax.set_ylabel("Z", rotation=0)
    fig.savefig(f"tests/test_scatter_tokamak_source_quantities_{quantity}.png")
    plt.close(fig)


@pytest.mark.parametrize("aspect", ["equal", "auto", 2])
def test_scatter_tokamak_source_aspect(tokamak_source, aspect):
    """Plot with various aspect ratios"""
    fig = plt.figure()
    ax = fig.gca()
    assert not ax.collections  # Check ax is empty
    ops_plt.scatter_tokamak_source(tokamak_source, ax=ax, aspect=aspect)
    assert ax.collections  # Check ax is not empty
    # Save for viewing, clean up
    ax.set_xlabel("R")
    ax.set_ylabel("Z", rotation=0)
    fig.savefig(f"tests/test_scatter_tokamak_source_aspect_{aspect}.png")
    plt.close(fig)


@pytest.mark.parametrize("kwargs", [{"alpha": 0.2}, {"marker": "x"}])
def test_scatter_tokamak_source_kwargs(tokamak_source, kwargs):
    """Plot with a kwarg compatible with 'scatter'"""
    fig = plt.figure()
    ax = fig.gca()
    assert not ax.collections  # Check ax is empty
    ops_plt.scatter_tokamak_source(tokamak_source, ax=ax, **kwargs)
    assert ax.collections  # Check ax is not empty
    # Save for viewing, clean up
    ax.set_xlabel("R")
    ax.set_ylabel("Z", rotation=0)
    fig.savefig(
        f"tests/test_scatter_tokamak_source_kwargs_{list(kwargs.keys())[0]}.png"
    )
    plt.close(fig)


def test_scatter_tokamak_not_source():
    """Ensure failure when given non-TokamakSource to plot"""
    with pytest.raises(ValueError) as excinfo:
        fig = plt.figure()
        ax = fig.gca()
        ops_plt.scatter_tokamak_source("hello world", ax=ax)
    plt.close()
    assert "TokamakSource" in str(excinfo.value)


@pytest.mark.parametrize("quantity", ["coucou", "ion_density", 17])
def test_scatter_tokamak_wrong_quantity(tokamak_source, quantity):
    """Ensure failure when incorrect quantity specified"""
    with pytest.raises(ValueError) as excinfo:
        fig = plt.figure()
        ax = fig.gca()
        ops_plt.scatter_tokamak_source(tokamak_source, ax=ax, quantity=quantity)
    plt.close()
    assert "quantity" in str(excinfo.value)


def test_plot_tokamak_source_3D_default(tokamak_source):
    """Ensure plots correctly with default inputs"""
    plt.figure()
    ops_plt.plot_tokamak_source_3D(tokamak_source)
    assert plt.gca().lines  # Check current ax is not empty
    # Save for viewing, clean up
    plt.savefig("tests/test_plot_tokamak_source_3D_defaults.png")
    plt.close()


def test_plot_tokamak_source_3D_with_ax(tokamak_source):
    """Ensure plots correctly given ax instance"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    assert not ax.lines  # Check ax is empty
    ops_plt.plot_tokamak_source_3D(tokamak_source, ax=ax)
    assert ax.lines  # Check ax is not empty
    # Save for viewing, clean up
    fig.savefig("tests/test_plot_tokamak_source_3D_with_ax.png")
    plt.close(fig)


@pytest.mark.parametrize(
    "quantity", ["ion_temperature", "neutron_source_density", "strength"]
)
def test_plot_tokamak_source_3D_quantities(tokamak_source, quantity):
    """Ensure plots correctly for each quantity"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    assert not ax.lines  # Check ax is empty
    ops_plt.plot_tokamak_source_3D(tokamak_source, ax=ax, quantity=quantity)
    assert ax.lines  # Check ax is not empty
    # Save for viewing, clean up
    fig.savefig(f"tests/test_plot_tokamak_source_3D_quantities_{quantity}.png")
    plt.close(fig)


@pytest.mark.parametrize("colorbar", ["plasma", "rainbow"])
def test_plot_tokamak_source_3D_colorbars(tokamak_source, colorbar):
    """Ensure plots correctly given colorbar choice"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    assert not ax.lines  # Check ax is empty
    ops_plt.plot_tokamak_source_3D(
        tokamak_source, ax=ax, quantity="ion_temperature", colorbar=colorbar
    )
    assert ax.lines  # Check ax is not empty
    # Save for viewing, clean up
    fig.savefig(f"tests/test_plot_tokamak_source_3D_colorbar_{colorbar}.png")
    plt.close(fig)


@pytest.mark.parametrize(
    "angles,name",
    [
        [(0, np.pi / 4), "eighth"],
        [(0, np.pi), "half"],
        [(np.pi, 2 * np.pi), "half_offset"],
        [(0, 2 * np.pi), "full"],
    ],
)
def test_plot_tokamak_source_3D_angles(tokamak_source, angles, name):
    """Ensure plots correctly given angles range"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    assert not ax.lines  # Check ax is empty
    ops_plt.plot_tokamak_source_3D(
        tokamak_source, ax=ax, quantity="ion_temperature", angles=angles
    )
    assert ax.lines  # Check ax is not empty
    # Save for viewing, clean up
    fig.savefig(f"tests/test_plot_tokamak_source_3D_angles_{name}.png")
    plt.close(fig)


@pytest.mark.parametrize("kwargs", [{"alpha": 0.2}, {"linestyle": "dotted"}])
def test_plot_tokamak_source_3D_kwargs(tokamak_source, kwargs):
    """Ensure plots correctly given additonal keyword arguments to pass on to plot"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    assert not ax.lines  # Check ax is empty
    ops_plt.plot_tokamak_source_3D(
        tokamak_source, ax=ax, quantity="ion_temperature", **kwargs
    )
    assert ax.lines  # Check ax is not empty
    # Save for viewing, clean up
    fig.savefig(
        f"tests/test_plot_tokamak_source_3D_kwargs_{list(kwargs.keys())[0]}.png"
    )
    plt.close(fig)


def test_plot_tokamak_source_3D_not_source():
    """Ensure failure when given non-TokamakSource to plot"""
    with pytest.raises(ValueError) as excinfo:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ops_plt.plot_tokamak_source_3D("hello world", ax=ax)
    plt.close()
    assert "TokamakSource" in str(excinfo.value)


@pytest.mark.parametrize("quantity", ["coucou", "ion_density", 17])
def test_plot_tokamak_source_3D_wrong_quantity(tokamak_source, quantity):
    """Ensure failure when incorrect quantity specified"""
    with pytest.raises(ValueError) as excinfo:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ops_plt.plot_tokamak_source_3D(tokamak_source, ax=ax, quantity=quantity)
    plt.close()
    assert "quantity" in str(excinfo.value)
