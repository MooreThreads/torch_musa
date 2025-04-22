# pylint: disable=missing-module-docstring
from .device_interface import register_interface_for_device, MusaInterface
from .trace_rules import *

register_interface_for_device("musa", MusaInterface)
