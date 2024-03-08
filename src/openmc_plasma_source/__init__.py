try:
    from importlib.metadata import PackageNotFoundError, version
except (ModuleNotFoundError, ImportError):
    from importlib_metadata import PackageNotFoundError, version
try:
    __version__ = version("openmc_plasma_source")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

__all__ = ["__version__"]

from .fuel_types import get_neutron_energy_distribution
from .point_source import fusion_point_source
from .ring_source import fusion_ring_source
from .tokamak_source import *
