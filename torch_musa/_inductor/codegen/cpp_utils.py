"""CPP Utils"""

import torch._inductor.codegen.cpp_utils

DEVICE_TO_ATEN = {"cpu": "at::kCPU", "cuda": "at::kCUDA", "musa": "at::kMUSA"}

torch._inductor.codegen.cpp_utils.DEVICE_TO_ATEN = DEVICE_TO_ATEN
