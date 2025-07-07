# -*- coding: utf-8 -*-
try:
    from .data_loader import DataLoader  # noqa: F401
    from .energy_calculator import calculate_energies  # noqa: F401
    from .error_estimator import calculate_errors  # noqa: F401
except ImportError:
    pass

from ._version_helper import get_version

__version__ = get_version()
