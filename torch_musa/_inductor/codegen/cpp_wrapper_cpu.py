# pylint: disable=W0613

"""MUSA CPP wrapper impl"""
# pylint: disable=W0613

from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu


class CppWrapperMusa(CppWrapperCpu):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    @staticmethod
    def get_device_include_path(device: str) -> str:
        header_path = (
            "#include <torch_musa/csrc/inductor/aoti_torch/generated/"
            "c_shim_musa.h>\n"
            "#include <torch_musa/csrc/inductor/aoti_runtime/utils_musa.h>"
        )
        return header_path

    def codegen_invoke_subgraph(self, *args, **kwargs):
        """
        Override abstract method from CppWrapperCpu.
        TODO: Implement MUSA-specific subgraph invocation logic.
        """
        raise NotImplementedError(
            "CppWrapperMusa.codegen_invoke_subgraph is not implemented yet."
        )
