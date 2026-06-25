"""Monkey patches for Inductor template heuristics."""

# pylint: disable=too-few-public-methods

__all__ = ["_apply_template_heuristics_patches"]

from torch._inductor.template_heuristics import registry as _registry
from torch._inductor.template_heuristics import triton as _triton


class MUSAConfigHeuristic(_triton.BaseConfigHeuristic):
    """
    Placeholder child class for MUSA specific overrides.
    """


class MUSAMMTemplateConfigHeuristic(_triton.MMTemplateConfigMixin, MUSAConfigHeuristic):
    """Standard MM template heuristic for MUSA."""


class MUSAAddMMTemplateConfigHeuristic(
    _triton.AddMMConfigMixin, MUSAMMTemplateConfigHeuristic
):
    """Addmm specific mixin for MUSA."""


class MUSAScaledMMTemplateConfigHeuristic(
    _triton.ScaledMMConfigMixin, MUSAConfigHeuristic
):
    """Scaled MM template heuristic for MUSA."""

    def __init__(self) -> None:
        super().__init__()
        self.mm_configs = self.scaled_mm_configs
        self.exhaustive_configs = self.scaled_mm_configs


class MUSAInt8MMTemplateConfigHeuristic(
    _triton.INT8MMTemplateConfigMixin, MUSAConfigHeuristic
):
    """Int8 MM template heuristic for MUSA."""

    def __init__(self) -> None:
        super().__init__()
        self.mm_configs = self.int8_mm_configs
        self.exhaustive_configs = self.int8_mm_configs


class MUSAMMPlusMMTemplateConfigHeuristic(
    _triton.MMPlusMMTemplateConfigMixin, MUSAConfigHeuristic
):
    """MM Plus MM template heuristic for MUSA."""

    def __init__(self) -> None:
        super().__init__()
        self.mm_configs = self.mm_plus_mm_configs
        self.exhaustive_configs = self.mm_plus_mm_configs


_TRITON_CLASS_EXPORTS = [
    MUSAConfigHeuristic,
    MUSAMMTemplateConfigHeuristic,
    MUSAAddMMTemplateConfigHeuristic,
    MUSAScaledMMTemplateConfigHeuristic,
    MUSAInt8MMTemplateConfigHeuristic,
    MUSAMMPlusMMTemplateConfigHeuristic,
]

_MUSA_TEMPLATE_HEURISTICS = [
    (
        _triton.mm_template.uid,
        "musa",
        None,
        MUSAMMTemplateConfigHeuristic,
    ),
    (
        _triton.bmm_template.uid,
        "musa",
        None,
        MUSAMMTemplateConfigHeuristic,
    ),
    (
        _triton.mm_template.uid,
        "musa",
        "addmm",
        MUSAAddMMTemplateConfigHeuristic,
    ),
    (
        _triton.bmm_template.uid,
        "musa",
        "baddbmm",
        MUSAAddMMTemplateConfigHeuristic,
    ),
    (
        _triton.mm_template.uid,
        "musa",
        "scaled_mm",
        MUSAScaledMMTemplateConfigHeuristic,
    ),
    (
        _triton.mm_template.uid,
        "musa",
        "int_mm",
        MUSAInt8MMTemplateConfigHeuristic,
    ),
    (
        _triton.mm_plus_mm_template.uid,
        "musa",
        None,
        MUSAMMPlusMMTemplateConfigHeuristic,
    ),
]


def _apply_template_heuristics_patches() -> None:
    for cls in _TRITON_CLASS_EXPORTS:
        setattr(_triton, cls.__name__, cls)

    for template_name, device_type, op_name, cls in _MUSA_TEMPLATE_HEURISTICS:
        _registry.register_template_heuristic(
            template_name,
            device_type,
            op_name=op_name,
        )(cls)

    heuristic_cache = getattr(_registry, "_HEURISTIC_CACHE", None)
    if heuristic_cache is not None:
        for key in list(heuristic_cache):
            if len(key) > 1 and key[1] == "musa":
                heuristic_cache.pop(key, None)
