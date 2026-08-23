"""Helpers for keeping optional SPARC features out of lightweight imports."""

from importlib import import_module
from typing import Any, MutableMapping


LazyImports = dict[str, tuple[str, str]]

_ALGORITHM_MODULES = frozenset({
    "kneed",
    "psutil",
    "segment_anything",
    "sklearn",
    "torch",
    "torchvision",
})


def resolve_attribute(
    namespace: MutableMapping[str, Any],
    package: str,
    name: str,
    imports: LazyImports,
) -> Any:
    """Resolve and cache a public attribute declared by a package initializer."""
    try:
        module_name, attribute_name = imports[name]
    except KeyError as error:
        raise AttributeError(f"module {package!r} has no attribute {name!r}") from error

    try:
        module = import_module(module_name, package)
    except ModuleNotFoundError as error:
        missing_root = (error.name or "").partition(".")[0]
        if missing_root in _ALGORITHM_MODULES:
            raise ModuleNotFoundError(
                f"{package}.{name} requires SPARC's optional algorithm "
                "dependencies. Install them with `pip install 'sparc[algorithm]'`."
            ) from error
        raise

    value = getattr(module, attribute_name)
    namespace[name] = value
    return value


def public_dir(namespace: MutableMapping[str, Any], imports: LazyImports) -> list[str]:
    """Return normal module names plus unresolved lazy public names."""
    return sorted(set(namespace) | set(imports))
