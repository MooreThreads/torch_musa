# pylint: disable=W0223

"""MUSA Template Buffer"""
# pylint: disable=abstract-method

from torch._inductor.ir import TemplateBuffer


class MUSATemplateBuffer(TemplateBuffer):
    """
    MUSA Template Buffer.
    """

    def __init__(
        self,
        layout,
        inputs,
        make_kernel_render,
        workspace_size: int,
        template: "MUSATemplate",  # type: ignore[name-defined]
    ):
        super().__init__(layout, inputs, make_kernel_render)
        # Global memory (in bytes) needed for this template.
        self.workspace_size = workspace_size
        self.template = template

    def get_workspace_size(self):
        return self.workspace_size if self.workspace_size is not None else 0
