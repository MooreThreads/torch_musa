"""
This module provides a function for dynamically loading functions from external modules.
"""

import importlib


def load_ext(name, funcs=None):
    if funcs is None:
        funcs = []
    if not isinstance(funcs, (list, tuple)):
        funcs = [funcs]
    ext = importlib.import_module("torch_musa." + name)
    for func in funcs:
        assert hasattr(ext, func), f"{func} not found in module {name}"
    return ext
