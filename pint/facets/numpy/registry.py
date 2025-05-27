"""
pint.facets.numpy.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import Any

from ..plain import GenericPlainRegistry
from .quantity import NumpyQuantity
from .unit import NumpyUnit


class GenericNumpyRegistry(GenericPlainRegistry):
    pass


class NumpyRegistry(GenericPlainRegistry):
    Quantity = NumpyQuantity[Any]
    Unit = NumpyUnit
