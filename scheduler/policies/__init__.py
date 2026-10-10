"""Pluggable policy registry with auto-discovery.

Usage::

    from scheduler.policies import get_policy, list_policies

    policy = get_policy("carbon_aware", alpha=0.5, scheduler_group="us_east")
    print(list_policies())  # ['carbon_aware', 'least_loaded', ...]

To add a custom policy, create a new ``.py`` file in this directory,
inherit from ``BasePolicy``, and decorate with ``@register_policy("name")``.
It will be discovered automatically — no other files need editing.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Type

from .common.base import BasePolicy, PolicyScope  # noqa: F401

_REGISTRY: Dict[str, Type[BasePolicy]] = {}


def register_policy(name: str):
    """Class decorator that registers a policy under *name*.

    Example::

        @register_policy("my_policy")
        class MyPolicy(BasePolicy):
            ...
    """
    def decorator(cls: Type[BasePolicy]):
        if name in _REGISTRY:
            raise ValueError(
                f"Policy name {name!r} already registered by "
                f"{_REGISTRY[name].__name__}"
            )
        if not issubclass(cls, BasePolicy):
            raise TypeError(f"{cls.__name__} must inherit from BasePolicy")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_policy(name: str = "least_loaded", **kwargs) -> BasePolicy:
    """Instantiate a registered policy by name.

    All *kwargs* are forwarded to the constructor.  ``BasePolicy``
    accepts ``scheduler_group``; subclasses accept whatever they
    need (e.g. ``alpha`` for CarbonAware).
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown policy {name!r}. Available: {sorted(_REGISTRY.keys())}"
        )
    return cls(**kwargs)


def list_policies() -> List[str]:
    """Return sorted names of all registered policies."""
    return sorted(_REGISTRY.keys())


# ── Auto-discovery: every module in this package IS a policy ────────────
#
# Flat by design: `ls scheduler/policies/` should be the list of policies and
# nothing else, so everything they share lives in common/ and discovery skips
# it. Previously the greedy ones sat in a subdirectory and the three ILP ones
# shared a single 792-line module, so neither the directory nor the file names
# told you what existed -- the registry was the only honest answer.
_pkg_path = Path(__file__).parent
for _info in pkgutil.iter_modules([str(_pkg_path)]):
    if _info.name == "common":
        continue
    importlib.import_module(f".{_info.name}", __package__)
