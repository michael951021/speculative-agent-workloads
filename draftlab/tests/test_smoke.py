"""Import smoke test: the toolchain's heavy deps must import on CPU."""

import importlib

HEAVY = ["jax", "torch", "triton", "ray"]


def test_package_imports():
    import draftlab  # noqa: F401


def test_heavy_deps_import():
    for mod in HEAVY:
        importlib.import_module(mod)
