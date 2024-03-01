try:
    from importlib.metadata import version, PackageNotFoundError
except (ModuleNotFoundError, ImportError):
    from importlib_metadata import version, PackageNotFoundError
try:
    __version__ = version("openmc_plasma_source")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

__all__ = ["__version__"]

from .tokamak_source import TokamakSource
from .ring_source import FusionRingSource
from .point_source import fusion_point_source
from .fuel_types import get_neutron_energy_distribution
