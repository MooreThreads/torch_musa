"""Imports the torch musa adaption facilities."""

import torch
try:
    import torch_musa._MUSAC
except ImportError as err:
    raise ImportError("Please try running Python from a different directory!") from err
