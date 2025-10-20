# pylint: disable=C0301, C0114, R1705
import torch

__all__ = ["version"]


def version():
    """
    Returns the version of the MCCL.


    This function returns a tuple containing the major, minor, and patch version numbers of the MCCL.
    The suffix is also included in the tuple if a version suffix exists.
    Returns:
        tuple: The version information of the MCCL.
    """
    ver = torch.musa._MUSAC._mccl_version()
    major = ver >> 32
    minor = (ver >> 16) & 65535
    patch = ver & 65535
    suffix = torch.musa._MUSAC._mccl_version_suffix().decode("utf-8")
    if suffix == "":
        return (major, minor, patch)
    else:
        return (major, minor, patch, suffix)
