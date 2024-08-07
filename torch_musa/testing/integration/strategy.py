"""Strategy definitions and implementations"""

from abc import ABC, abstractmethod

import torch

# pylint: disable=unused-import
import torch_musa


class GlobalStrategy(ABC):
    """
    Global strategy uses python contextlib to manage the enablement/restoration
    of global settings on the framework side(e.g. tf32). It's lifecycle includes
    the entire training/evaluation process for integration testing.
    """

    @abstractmethod
    def make_context(self): ...


class ModelStrategy(ABC):
    """
    Model strategy completes the model conversion(e.g. to channels_last format)
    before the training/evaluation process. The transformed model will be applied
    throughout the entire integration testing without any restoration.
    """

    @abstractmethod
    def transform(self, model: torch.nn.Module) -> torch.nn.Module: ...


class RuntimeStrategy(ABC):
    """
    Runtime strategy takes effect in the following two scenarios:
    1. During the training stage, exclude the subsequent testing stage in training process.
    2. Entire evaluation process.
    For example, in the training stage we can apply AMP(Automatic mixed-precision) for
    acceleration, while in the inference stage we can directly convert the model to fp16.
    """

    @abstractmethod
    def make_context(self): ...


class TF32(GlobalStrategy):
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def make_context(self):
        return torch.backends.mudnn.flags(self.enabled)


class AMP(RuntimeStrategy):
    """Automatic mixed-precision for training"""

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.dtype = dtype
        self.cache_enabled = cache_enabled

    def make_context(self):
        return torch.musa.amp.autocast(self.enabled, self.dtype, self.cache_enabled)


class ChannelsLast2D(ModelStrategy):
    def __init__(self) -> None:
        pass

    def transform(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(memory_format=torch.channels_last)
        return model
