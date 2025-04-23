"""torch_musa ao"""

__all__ = [
    "nn",
]


def __getattr__(name):
    if name in __all__:
        import importlib  # pylint: disable=C0415

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
