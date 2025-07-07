# -*- coding: utf-8 -*-
from pathlib import Path


def get_version():
    if (Path(__file__).parent / "_version.py").exists():
        from ._version import __version__  # noqa F401
    else:
        __version__ = "0.0.1"
    return __version__
