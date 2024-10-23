from enum import Enum, auto

from torchgen.model import (
    DispatchKey,
)


class OpKind(Enum):
    ONLY_CPU = auto()
    CUDA_CUSTOM = auto()
    ONLY_COMPOSITE = auto()
    SAME_CPU_CUDA = auto()
    OTHERS = auto()

    def description(self) -> str:
        if self is OpKind.ONLY_CPU:
            return "Explicitly registered only by CPU kernel, missing CUDA dispatch."
        elif self is OpKind.CUDA_CUSTOM:
            return (
                "CUDA customized kernel, CPU dispatch function is different or missing."
            )
        elif self is OpKind.ONLY_COMPOSITE:
            return "Composite kernel entrypoint, no CPU/CUDA specialized dispatches."
        elif self is OpKind.SAME_CPU_CUDA:
            return "CPU/CUDA share the same kernel explicitly."
        return (
            "Temporarily ignored by torch_musa, may further be processed in the future."
        )

    @classmethod
    def choices(cls):
        return [key.lower() for key in cls.__members__.keys()]

    @classmethod
    def parse(cls, text):
        assert text in cls.choices()
        return cls.__members__[text.upper()]

    def is_exportable(self) -> bool:
        return (self is OpKind.CUDA_CUSTOM) or (self is OpKind.SAME_CPU_CUDA)

    def binding_impl(self, dispatch):
        assert self.is_exportable()
        return dispatch[DispatchKey.CUDA]
