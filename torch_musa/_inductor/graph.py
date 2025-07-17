"""AOTI Core Impl: MusaGraphLowering Class"""

from typing import (
    Union,
    Tuple,
    Dict,
)
import logging

import torch
from torch._inductor.graph import GraphLowering, getattr_recursive
from torch._inductor import ir
from torch._inductor.ir import Constant, TensorBox, TorchBindObject
from torch._inductor.lowering import unsupported_output_tensor, tensor
from torch.utils._mode_utils import no_dispatch

from torch._inductor.codegen.common import (
    get_wrapper_codegen_for_device,
    get_device_op_overrides,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo.utils import defake
from torch._inductor.graph import output_code_log
from torch._inductor import config
from torch._inductor.virtualized import V
from torch._inductor.scheduler import Scheduler

from .codegen.cpp_wrapper_musa import MUSACppWrapper
from .codegen.cpp_wrapper_cpu import CppWrapperMusa
from .codegen.codecache import AotCodeCompiler


log = logging.getLogger(__name__)


class MusaGraphLowering(GraphLowering):
    """MUSA Graph Lowering class"""

    def init_wrapper_code(self):
        self.musa = "musa" in self.device_types
        if self.cpp_wrapper:
            self.validate_can_generate_cpp_wrapper()
            self.wrapper_code = MUSACppWrapper()
            if self.musa:
                self.wrapper_code = MUSACppWrapper()
            else:
                self.wrapper_code = CppWrapperMusa()
            return

        device_types = self.device_types.copy()
        device_types.discard("cpu")
        # TODO(Eikan): Only support mixing cpu and other device now.
        assert len(device_types) <= 1, f"Does not support mixing {device_types}"
        only_cpu = len(device_types) == 0
        device_type = "cpu" if only_cpu else device_types.pop()

        self.device_ops = get_device_op_overrides(device_type)
        wrapper_code_gen_cls = get_wrapper_codegen_for_device(device_type)
        assert wrapper_code_gen_cls is not None, f"Device {device_type} not supported"
        self.wrapper_code = wrapper_code_gen_cls()

    def codegen_with_cpp_wrapper(self):
        """
        For CPU, the cpp wrapper codegen is done in one pass.
        For GPU, the cpp wrapper codegen is done in two steps: JIT-compile the model with python
        wrapper code and run it to generate autotuned kernel binaries in the first pass; and then
        generate cpp wrapper code and compile it to a dynamic library in the second pass.
        """
        if "musa" in self.device_types:
            # first pass
            self.cpp_wrapper = False
            compiled = self.compile_to_module().call

            def materialize(x):
                if isinstance(x, (torch.SymInt, torch.SymFloat)):
                    # Need concrete value to run dynamic shapes and tune the result
                    return x.node.hint
                if isinstance(x, FakeTensor):
                    return defake(x)
                assert isinstance(
                    x, torch.Tensor
                ), "Unknown type when creating real inputs" + str(type(x))
                return x

            with torch.utils._python_dispatch._disable_current_modes():
                assert self.example_inputs is not None
                real_inputs = [materialize(x) for x in self.example_inputs]
                compiled(real_inputs)
            del real_inputs

            # second pass
            # TODO: reuse self.scheduler from the first pass to speed up the second pass
            self.cpp_wrapper = True
            self.removed_buffers.clear()
            self.inplaced_to_remove.clear()
            return self.codegen()
        # cpu
        return self.codegen()

    def compile_to_fn(self):
        if self.aot_mode:

            assert self.cpp_wrapper, "AOT mode only supports C++ wrapper"
            code, _ = self.codegen_with_cpp_wrapper()
            output_code_log.debug("Output code: \n%s", code)

            serialized_extern_kernel_nodes = None
            if (
                config.is_fbcode()
                and self.extern_kernel_nodes
                and self.extern_node_serializer
            ):
                serialized_extern_kernel_nodes = self.extern_node_serializer(
                    self.extern_kernel_nodes
                )
                output_code_log.debug(
                    "Serialized Extern Kernel Nodes: \n%s",
                    serialized_extern_kernel_nodes,
                )

            # Directly return the file path with the compiled code
            return AotCodeCompiler.compile(
                self, code, serialized_extern_kernel_nodes, musa=self.musa
            )
        # not aot mode
        return self.compile_to_module().call

    def codegen(self):

        self.init_wrapper_code()

        self.scheduler = Scheduler(self.buffers)
        V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)

        self.wrapper_code.push_codegened_graph(self)
        self.scheduler.codegen()
        log.debug(
            "Finished codegen for all nodes. The list of kernel names available: %s",
            V.graph.all_codegen_kernel_names,
        )

        result = self.wrapper_code.generate(self.is_inference)
        self.wrapper_code.pop_codegened_graph()
        return result

    # fix torch.compile constant has no attr get_name
    # url: https://github.com/pytorch/pytorch/pull/141226/files
    def get_attr(
        self, target: str, args: Tuple[()], kwargs: Dict[str, object]  # type: ignore[override]
    ) -> Union[Constant, TensorBox, ir.Subgraph, TorchBindObject]:
        # this is a constant
        value = getattr_recursive(self.module, target)  # type: ignore[arg-type]

        if isinstance(value, torch.fx.GraphModule):
            return ir.Subgraph(name=target, graph_module=value)

        if isinstance(value, torch._C.ScriptObject):
            self.torchbind_constants[target] = value
            self.constant_reprs[target] = ""
            return TorchBindObject(target, value)

        assert isinstance(value, torch.Tensor)
        if (
            config.aot_inductor.use_runtime_constant_folding
            or config.always_keep_tensor_constants
            or unsupported_output_tensor(value)
        ):
            return self.add_tensor_constant(value, target)

        with no_dispatch():
            # if value.shape == ():
            #     return Constant(value.item(), value.dtype, value.device)
            if self.can_inline_constant(value):
                log.debug("Inlining constant: %s ", str(target))
                # tensor lowering has constant inlining logic

                return tensor(value.tolist(), dtype=value.dtype, device=value.device)

        return self.add_tensor_constant(value, target)
