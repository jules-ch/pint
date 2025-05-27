"""
pint.registry
~~~~~~~~~~~~~

Defines the UnitRegistry, a class to contain units and their relations.

This registry contains all pint capabilities, but you can build your
customized registry by picking only the features that you actually
need.

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, overload

from pint._typing import M, PreProcessorCallable, ScalarT, UnitLike
from pint.facets.context.objects import ContextQuantity, ContextUnit
from pint.facets.dask import DaskQuantity, DaskUnit
from pint.facets.measurement.objects import MeasurementQuantity, MeasurementUnit
from pint.facets.nonmultiplicative.objects import (
    NonMultiplicativeQuantity,
    NonMultiplicativeUnit,
)
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.numpy.unit import NumpyUnit
from pint.facets.plain.quantity import PlainQuantity
from pint.facets.plain.registry import NON_INT_TYPE
from pint.facets.plain.unit import PlainUnit
from pint.facets.system.objects import SystemQuantity, SystemUnit

from . import facets, registry_helpers
from .util import logger, pi_theorem

# To build the Quantity and Unit classes
# we follow the UnitRegistry bases
# but


class Quantity(
    SystemQuantity[M],
    ContextQuantity[M],
    DaskQuantity[M],
    NumpyQuantity[M],
    MeasurementQuantity[M],
    NonMultiplicativeQuantity[M],
    PlainQuantity[M],
    Generic[M],
):
    @overload
    def __new__(cls, value: str, units: UnitLike | None = None) -> Quantity[Any]: ...

    @overload
    def __new__(  # type: ignore[misc]
        cls, value: Sequence[ScalarT], units: UnitLike | None = None
    ) -> Quantity[Any]: ...

    @overload
    def __new__(
        cls, value: Quantity[Any], units: UnitLike | None = None
    ) -> Quantity[Any]: ...
    @overload
    def __new__(cls, value: M, units: UnitLike | None = None) -> Quantity[M]: ...

    def __new__(
        cls, value: M | Sequence[Any] | Quantity[Any], units: UnitLike | None = None
    ) -> Quantity[Any]:
        return super().__new__(cls, value, units)


class Unit(
    SystemUnit,
    ContextUnit,
    DaskUnit,
    NumpyUnit,
    MeasurementUnit,
    NonMultiplicativeUnit,
    PlainUnit,
):
    pass


class GenericUnitRegistry(
    facets.GenericSystemRegistry,
    facets.GenericContextRegistry,
    facets.GenericDaskRegistry,
    facets.GenericNumpyRegistry,
    facets.GenericMeasurementRegistry,
    facets.GenericNonMultiplicativeRegistry,
    facets.GenericPlainRegistry,
):
    pass


class UnitRegistry(GenericUnitRegistry):
    """The unit registry stores the definitions and relationships between units.

    Parameters
    ----------
    filename :
        path of the units definition file to load or line-iterable object.
        Empty string to load the default definition file. (default)
        None to leave the UnitRegistry empty.
    force_ndarray : bool
        convert any input, scalar or not to a numpy.ndarray.
        (Default: False)
    force_ndarray_like : bool
        convert all inputs other than duck arrays to a numpy.ndarray.
        (Default: False)
    default_as_delta :
        In the context of a multiplication of units, interpret
        non-multiplicative units as their *delta* counterparts.
        (Default: False)
    autoconvert_offset_to_baseunit :
        If True converts offset units in quantities are
        converted to their plain units in multiplicative
        context. If False no conversion happens. (Default: False)
    on_redefinition : str
        action to take in case a unit is redefined.
        'warn', 'raise', 'ignore' (Default: 'raise')
    auto_reduce_dimensions :
        If True, reduce dimensionality on appropriate operations.
        (Default: False)
    autoconvert_to_preferred :
        If True, converts preferred units on appropriate operations.
        (Default: False)
    preprocessors :
        list of callables which are iteratively ran on any input expression
        or unit string or None for no preprocessor.
        (Default=None)
    fmt_locale :
        locale identifier string, used in `format_babel` or None.
        (Default=None)
    case_sensitive : bool, optional
        Control default case sensitivity of unit parsing. (Default: True)
    cache_folder : str or pathlib.Path or None, optional
        Specify the folder in which cache files are saved and loaded from.
        If None, the cache is disabled. (default)
    """

    Quantity = Quantity
    Unit = Unit

    def __init__(
        self,
        filename: str = "",
        force_ndarray: bool = False,
        force_ndarray_like: bool = False,
        default_as_delta: bool = True,
        autoconvert_offset_to_baseunit: bool = False,
        on_redefinition: str = "warn",
        system: str | None = None,
        auto_reduce_dimensions: bool = False,
        autoconvert_to_preferred: bool = False,
        preprocessors: list[PreProcessorCallable] | None = None,
        fmt_locale: str | None = None,
        non_int_type: NON_INT_TYPE = float,
        case_sensitive: bool = True,
        cache_folder: str | Path | None = None,
    ):
        super().__init__(
            filename=filename,
            force_ndarray=force_ndarray,
            force_ndarray_like=force_ndarray_like,
            on_redefinition=on_redefinition,
            default_as_delta=default_as_delta,
            autoconvert_offset_to_baseunit=autoconvert_offset_to_baseunit,
            system=system,
            auto_reduce_dimensions=auto_reduce_dimensions,
            autoconvert_to_preferred=autoconvert_to_preferred,
            preprocessors=preprocessors,
            fmt_locale=fmt_locale,
            non_int_type=non_int_type,
            case_sensitive=case_sensitive,
            cache_folder=cache_folder,
        )

    def pi_theorem(self, quantities):
        """Builds dimensionless quantities using the Buckingham π theorem

        Parameters
        ----------
        quantities : dict
            mapping between variable name and units

        Returns
        -------
        list
            a list of dimensionless quantities expressed as dicts

        """
        return pi_theorem(quantities, self)

    def setup_matplotlib(self, enable: bool = True) -> None:
        """Set up handlers for matplotlib's unit support.

        Parameters
        ----------
        enable : bool
            whether support should be enabled or disabled (Default value = True)

        """
        # Delays importing matplotlib until it's actually requested
        from .matplotlib import setup_matplotlib_handlers

        setup_matplotlib_handlers(self, enable)

    wraps = registry_helpers.wraps

    check = registry_helpers.check


class LazyRegistry:
    def __init__(self, args=None, kwargs=None):
        self.__dict__["params"] = args or (), kwargs or {}

    def __init(self):
        args, kwargs = self.__dict__["params"]
        kwargs["on_redefinition"] = "raise"
        self.__class__ = UnitRegistry
        self.__init__(*args, **kwargs)
        self._after_init()

    def __getattr__(self, item):
        if item == "_on_redefinition":
            return "raise"
        self.__init()
        return getattr(self, item)

    def __setattr__(self, key, value):
        if key == "__class__":
            super().__setattr__(key, value)
        else:
            self.__init()
            setattr(self, key, value)

    def __getitem__(self, item):
        self.__init()
        return self[item]

    def __call__(self, *args, **kwargs):
        self.__init()
        return self(*args, **kwargs)


class ApplicationRegistry:
    """A wrapper class used to distribute changes to the application registry."""

    __slots__ = ["_registry"]

    def __init__(self, registry):
        self._registry = registry

    def get(self):
        """Get the wrapped registry"""
        return self._registry

    def set(self, new_registry):
        """Set the new registry

        Parameters
        ----------
        new_registry : ApplicationRegistry or LazyRegistry or UnitRegistry
            The new registry.

        See Also
        --------
        set_application_registry
        """
        if isinstance(new_registry, type(self)):
            new_registry = new_registry.get()

        if not isinstance(new_registry, (LazyRegistry, UnitRegistry)):
            raise TypeError("Expected UnitRegistry; got %s" % type(new_registry))
        logger.debug(
            "Changing app registry from %r to %r.", self._registry, new_registry
        )
        self._registry = new_registry

    def __getattr__(self, name):
        return getattr(self._registry, name)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            setattr(self._registry, name, value)

    def __dir__(self):
        return dir(self._registry)

    def __getitem__(self, item):
        return self._registry[item]

    def __call__(self, *args, **kwargs):
        return self._registry(*args, **kwargs)

    def __contains__(self, item):
        return self._registry.__contains__(item)

    def __iter__(self):
        return iter(self._registry)
