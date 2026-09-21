"""AOT Inductor Unitest"""

# Owner(s): ["module: inductor"]
import copy
import itertools
import os
import sys
import tempfile
import pytest
import types
import unittest
import dataclasses
from typing import Dict, Tuple
from unittest import skip
from unittest.mock import patch

import torch
import torch._export
import torch._inductor
from torch import nn
from torch._dynamo.testing import rand_strided, same
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import TestCase
from torch.export import Dim
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_quantization import (
    skip_if_no_torchvision,
    skipIfNoFBGEMM,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    IS_MACOS,
)

from torch.utils import _pytree as pytree

import torch_musa
from torch_musa import testing
from torch_musa.testing.base_test_tool import _HAS_TRITON

# pylint: disable-all


@dataclasses.dataclass
class TestFailure:
    suffixes: Tuple[str]
    is_skip: bool = False
    __test__: bool = False


try:
    from test_aot_inductor_utils import (
        AOTIRunnerUtil,
        prepend_counters,
        WhileLoopModels,
    )
    from test_aot_inductor_package import copy_tests

except (unittest.SkipTest, ImportError) as e:
    if __name__ == "__main__":
        sys.exit(0)
    raise


def check_model(
    self: TestCase,
    model,
    example_inputs,
    options=None,
    dynamic_shapes=None,
    disable_constraint_solver=False,
    atol=None,
    rtol=None,
):
    with torch.no_grad(), config.patch(
        {
            # "abi_compatible": self.abi_compatible,
            # "allow_stack_allocation": self.allow_stack_allocation,
            # "use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
        }
    ):
        torch.manual_seed(0)
        if not isinstance(model, types.FunctionType):
            model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(example_inputs)
        expected = ref_model(*ref_inputs)

        torch.manual_seed(0)
        actual = AOTIRunnerUtil.run(
            self.device,
            model,
            example_inputs,
            options,
            dynamic_shapes,
            disable_constraint_solver,
        )

    self.assertEqual(actual, expected, atol=atol, rtol=rtol)


def check_model_with_multiple_inputs(
    self: TestCase,
    model,
    list_example_inputs,
    options=None,
    dynamic_shapes=None,
):
    with torch.no_grad(), config.patch(
        {
            # "abi_compatible": self.abi_compatible,
            # "allow_stack_allocation": self.allow_stack_allocation,
        }
    ):
        torch.manual_seed(0)
        model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(list_example_inputs)
        list_expected = [ref_model(*inputs) for inputs in ref_inputs]

        torch.manual_seed(0)
        list_actual = AOTIRunnerUtil.run_multiple(
            self.device, model, list_example_inputs, options, dynamic_shapes
        )

    self.assertTrue(same(list_actual, list_expected))


def code_check_count(
    self: TestCase,
    model,
    example_inputs,
    target_str: str,
    target_count: int,
):
    so_path = torch._export.aot_compile(model, example_inputs)
    with open(os.path.splitext(so_path)[0] + ".cpp") as cpp:
        src_code = cpp.read()
        FileCheck().check_count(
            target_str,
            target_count,
            exactly=True,
        ).run(src_code)


class AOTInductorTestsTemplate:
    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_simple(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(16, 16, device=self.device),
            torch.randn(16, 16, device=self.device),
        )
        self.check_model(Model(), example_inputs, atol=1e-3, rtol=1e-3)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_constant_folding(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.w_pre = torch.randn(4, 4, device=device)
                self.b = torch.randn(4, device=device)

            def forward(self, x):
                w_transpose = torch.transpose(self.w_pre, 0, 1)
                w_relu = torch.nn.functional.relu(w_transpose)
                w = w_relu + self.b
                return torch.matmul(x, w)

        example_inputs = (torch.randn(4, 4, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_conv(self):
        for dtype, groups in itertools.product([torch.bfloat16, torch.float], [1, 2]):
            iC = 2
            oC = 3

            class Model(torch.nn.Module):
                def __init__(self, device):
                    super().__init__()
                    self.weight = torch.randn(oC * groups, iC, 3, 3, device=device).to(
                        dtype
                    )

                def forward(self, y):
                    return torch.nn.functional.conv2d(y, self.weight, groups=groups)

            example_inputs = (
                torch.randn(2, iC * groups, 10, 10, device=self.device).to(dtype),
            )

            with config.patch({}):
                self.check_model(Model(self.device), example_inputs)

    # TODO: current triton has issue with layernorm
    # @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    # def test_seq(self):
    #     layernorm = torch.nn.LayerNorm(16)
    #     net = torch.nn.Sequential(
    #         layernorm,
    #         torch.nn.ReLU(),
    #         layernorm,
    #         torch.nn.ReLU(),
    #     )

    #     example_inputs = (torch.randn(16, device=self.device),)
    #     self.check_model(net.eval(), example_inputs, atol=1e-3, rtol=1e-3)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_addmm(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 8
        N = 6
        K = 16
        model = Model(N, K, self.device)
        batch = 2
        a = torch.randn(batch, M, K, device=self.device)
        example_inputs = (a,)
        self.check_model(model, example_inputs)

    # @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    # def test_addmm_multiple_dynamic(self):
    #     class Model(torch.nn.Module):
    #         def __init__(self, n, k, device):
    #             super().__init__()
    #             self.weight = torch.randn(n, k, device=device)
    #             self.bias = torch.randn(n, device=device)

    #         def forward(self, a):
    #             return torch.nn.functional.linear(a, self.weight, self.bias)

    #     M = 8
    #     N = 6
    #     K = 16
    #     model = Model(N, K, self.device)
    #     batch = 2
    #     a = torch.randn(batch, M, K, device=self.device)
    #     dim0_a = Dim("dim0_a", min=1, max=2048)
    #     dynamic_shapes = {"a": {0: dim0_a}}
    #     list_example_inputs = [(a,)]
    #     batch = 2048
    #     list_example_inputs.append(
    #         (torch.randn(batch, M, K, device=self.device),),
    #     )
    #     batch = 128
    #     list_example_inputs.append(
    #         (torch.randn(batch, M, K, device=self.device),),
    #     )
    #     self.check_model_with_multiple_inputs(
    #         model,
    #         list_example_inputs,
    #         dynamic_shapes=dynamic_shapes,
    #         options={
    #             "max_autotune": True,
    #             "max_autotune_gemm_backends": "TRITON",
    #         },
    #     )

    @skipIfNoFBGEMM
    def test_quanatized_int8_linear(self):
        class Model(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.weight = torch.randn(10, 10, device=device)
                self.bias = torch.randn(10, device=device)
                self.input_scale = torch.tensor(0.1)
                self.input_zero_point = torch.tensor(0)
                self.weight_scale = torch.tensor(0.1)
                self.weight_zero_point = torch.tensor(0)
                self.output_scale = torch.tensor(0.1)
                self.output_zero_point = torch.tensor(0)
                self.out_channel = 10

            def forward(self, x):
                return torch.ops._quantized.wrapped_quantized_linear(
                    x,
                    self.input_scale,
                    self.input_zero_point,
                    self.weight,
                    self.weight_scale,
                    self.weight_zero_point,
                    self.bias,
                    self.output_scale,
                    self.output_zero_point,
                    self.out_channel,
                )

        example_inputs = (torch.randn(10, 10, device=self.device),)
        with config.patch({"aot_inductor.use_runtime_constant_folding": True}):
            self.check_model(Model(self.device), example_inputs)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_while_loop_simple(self):
        inputs = (
            torch.randn((10, 20), device=self.device),
            torch.randn((10, 20), device=self.device),
        )
        dim0_ab = Dim("s0", min=2, max=1024)
        dynamic_shapes = {
            "ci": {},
            "a": {0: dim0_ab, 1: None},
            "b": {0: dim0_ab, 1: None},
        }
        self.check_model_with_multiple_inputs(
            WhileLoopModels.Simple(),
            prepend_counters(inputs),
            dynamic_shapes=dynamic_shapes,
        )

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    @config.patch({"is_predispatch": True})
    def test_constant(self):
        class M(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device

            def forward(self, x):
                t = torch.tensor(x.size(-1), device=self.device, dtype=torch.float)
                t = torch.sqrt(t * 3)
                return x * t

        self.check_model(M(self.device), (torch.randn(5, 5, device=self.device),))

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_reuse_kernel(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                a = torch.sin(x)
                b = torch.mm(a, y)
                c = torch.sin(b)
                d = torch.mm(b, c)
                return d

        example_inputs = (
            torch.randn(87, 87, device=self.device),
            torch.randn(87, 87, device=self.device),
        )
        model = Model()
        self.check_model(
            model, example_inputs, atol=1e-3, rtol=1e-3
        )  # 1e-4 is the tol value used in pytorch/torch/_dynamo/utils.py

        if self.device == "musa":
            self.code_check_count(
                model, example_inputs, "triton_poi_fused_sin_0 = loadKernel(", 1
            )

    # @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    # def test_consecutive_compiles(self):
    #     """Test that compilation behaves correctly with cache hits"""

    #     class TestModule(torch.nn.Module):
    #         def __init__(self) -> None:
    #             super().__init__()

    #         def forward(self, x):
    #             return x + 1

    #     mod = TestModule()
    #     inp = torch.rand(1)
    #     mod(inp)
    #     mod2 = torch.fx.symbolic_trace(mod, concrete_args=[inp])
    #     so = torch._export.aot_compile(mod2, (inp,))
    #     assert so is not None
    #     # compile the 2nd time with cache hit
    #     so = torch._export.aot_compile(mod2, (inp,))
    #     assert so is not None

    # @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    # def test_empty_graph(self):
    #     class Model(torch.nn.Module):
    #         def __init__(self) -> None:
    #             super().__init__()

    #         def forward(self, x):
    #             return x

    #     example_inputs = (torch.randn(8, 4, 4, device=self.device),)
    #     self.check_model(Model(), example_inputs)

    # @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    # def test_convolution(self):
    #     class Model(torch.nn.Module):
    #         def __init__(self) -> None:
    #             super().__init__()

    #         def forward(self, x, w, b):
    #             return torch.ops.aten.convolution(x, w, b, [4], [0], [1], True, [0], 1)

    #     example_inputs = (
    #         torch.randn([2, 32, 90], device=self.device),
    #         torch.randn([32, 16, 8], device=self.device),
    #         torch.randn([16], device=self.device),
    #     )
    #     with config.patch(
    #         {
    #             "max_autotune": True,
    #             "max_autotune_gemm_backends": "Triton",
    #         }
    #     ):
    #         self.check_model(Model(), example_inputs)

    # @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    # def test_fqn(self):
    #     class NestedChild(torch.nn.Module):
    #         def __init__(self) -> None:
    #             super().__init__()
    #             self.nestedchild3buffer = torch.nn.Buffer(torch.ones(2, 3) * 3)

    #         def forward(self, x):
    #             return x / self.nestedchild3buffer

    #     class Child1(torch.nn.Module):
    #         def __init__(self) -> None:
    #             super().__init__()
    #             self.nested = NestedChild()
    #             self.register_parameter(
    #                 "child1param", torch.nn.Parameter(torch.ones(2, 3))
    #             )

    #         def forward(self, x):
    #             x = self.nested(x)
    #             return x + self.child1param

    #     class Child2(torch.nn.Module):
    #         def __init__(self) -> None:
    #             super().__init__()
    #             self.child2buffer = torch.nn.Buffer(torch.ones(2, 3) * 2)

    #         def forward(self, x):
    #             return x - self.child2buffer

    #     class MyModule(torch.nn.Module):
    #         def __init__(self) -> None:
    #             super().__init__()
    #             self.foo = Child1()
    #             self.bar = Child2()
    #             self.register_parameter(
    #                 "rootparam", torch.nn.Parameter(torch.ones(2, 3) * 4)
    #             )

    #         def forward(self, x):
    #             x = x * self.rootparam
    #             x = self.foo(x)
    #             x = self.bar(x)
    #             return x

    #     orig_eager = MyModule()

    #     self.check_model(MyModule(), (torch.randn(2, 3, device=self.device),), atol=1e-3, rtol=1e-3)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_model_modified_weights(self):
        class Model(torch.nn.Module):
            def __init__(self, n, k, device):
                super().__init__()
                self.weight = torch.randn(n, k, device=device)
                self.bias = torch.randn(n, device=device)

            def forward(self, a):
                return torch.nn.functional.linear(a, self.weight, self.bias)

        M = 16
        N = 10
        K = 128
        batch = 8
        example_inputs = (torch.randn(2, M, K, device=self.device),)
        model = Model(N, K, self.device)
        self.check_model(model, example_inputs)
        # Update model weights, after this AOTInductor should re-generate model.so
        # if weights are stored in the model.so
        model.weight += 1
        self.check_model(model, example_inputs)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    @common_utils.parametrize("max_autotune", [False, True])
    def test_misc_1(self, max_autotune):
        if self.device == "cpu" and IS_MACOS and max_autotune:
            raise unittest.SkipTest("max_autotune not supported on macos")

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.Sigmoid()
                )
                self.emb = nn.EmbeddingBag(num_embeddings=128, embedding_dim=32)
                self.over_arch = nn.Sequential(
                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 32), nn.Sigmoid()
                )

            def forward(self, x, y):
                mlp_output = self.mlp(x)
                emb_output = self.emb(y)
                return self.over_arch(torch.concat([mlp_output, emb_output], dim=1))

        example_inputs = (
            torch.randn(16, 128, device=self.device),
            torch.randint(0, 128, (16, 10), device=self.device),
        )
        self.check_model(
            Model(), example_inputs, options=dict(max_autotune=max_autotune)
        )


common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)


class AOTITestCase(TestCase):
    pass


def fail_musa(is_skip=False):
    return TestFailure(
        ("abi_compatible_musa", "non_abi_compatible_musa"),
        is_skip=is_skip,
    )


def fail_abi_compatible_musa(is_skip=False):
    return TestFailure(
        ("abi_compatible_musa",),
        is_skip=is_skip,
    )


def fail_non_abi_compatible_musa(is_skip=False):
    return TestFailure(
        ("non_abi_compatible_musa",),
        is_skip=is_skip,
    )


# test_failures, xfail by default, set is_skip=True to skip
MUSA_TEST_FAILURES = {
    # quantized unsupported for GPU
    "test_quanatized_int8_linear": fail_musa(is_skip=True),
}


class AOTInductorTestABICompatibleMusa(AOTITestCase):
    device = "musa"
    abi_compatible = True
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleMusa,
    "abi_compatible_musa",
    MUSA_TEST_FAILURES,
)


class AOTInductorTestNonABICompatibleMusa(AOTITestCase):
    device = "musa"
    abi_compatible = False
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestNonABICompatibleMusa,
    "non_abi_compatible_musa",
    MUSA_TEST_FAILURES,
)


class AOTInductorDynamicShapeTest(TestCase):
    """Regression tests for AOTI dynamic-batch specialization bugs."""

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_convolution_dynamic_batch(self):
        # Inductor keeps convolution as an ATen fallback node. AOTI should
        # lower it to the direct MUSA convolution C-shim instead of using the
        # ProxyExecutor fallback.
        class Conv(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.nn.functional.conv2d(
                    x,
                    weight,
                    bias,
                    stride=1,
                    padding=1,
                    dilation=1,
                    groups=1,
                )

        dev = "musa"
        batch = Dim("batch", min=1, max=64)
        inputs = (
            torch.randn(8, 3, 16, 16, dtype=torch.float16, device=dev),
            torch.randn(4, 3, 3, 3, dtype=torch.float16, device=dev),
            torch.randn(4, dtype=torch.float16, device=dev),
        )
        ep = torch.export.export(
            Conv().eval(),
            inputs,
            dynamic_shapes=({0: batch}, {}, {}),
            strict=False,
        )
        with patch("torch._inductor.ir.log.warning") as warning:
            pkg = torch._inductor.aoti_compile_and_package(ep)
        proxy_fallback_warnings = [
            call
            for call in warning.call_args_list
            if call.args and "missing a c-shim implementation" in str(call.args[0])
        ]
        self.assertFalse(proxy_fallback_warnings)
        runner = torch._inductor.aoti_load_package(pkg)

        for b in [1, 8, 16, 32, 64]:
            x = torch.randn(b, 3, 16, 16, dtype=torch.float16, device=dev)
            out = runner(x, inputs[1], inputs[2])
            out = out[0] if isinstance(out, (list, tuple)) else out
            ref = Conv().eval()(x, inputs[1], inputs[2])
            self.assertEqual(out.shape, ref.shape)
            torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_layer_norm_dynamic_batch(self):
        # Inductor decomposes layer_norm into reduction and pointwise Triton
        # kernels. AOTI should launch the generated MUSA Triton kernels with
        # dynamic batch sizes and optional weight and bias inputs.
        class LayerNorm(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.nn.functional.layer_norm(
                    x,
                    (16,),
                    weight,
                    bias,
                    eps=1e-5,
                )

        dev = "musa"
        batch = Dim("batch", min=1, max=64)
        inputs = (
            torch.randn(8, 4, 16, dtype=torch.float16, device=dev),
            torch.randn(16, dtype=torch.float16, device=dev),
            torch.randn(16, dtype=torch.float16, device=dev),
        )
        ep = torch.export.export(
            LayerNorm().eval(),
            inputs,
            dynamic_shapes=({0: batch}, {}, {}),
            strict=False,
        )
        pkg = torch._inductor.aoti_compile_and_package(ep)
        runner = torch._inductor.aoti_load_package(pkg)

        for b in [1, 8, 16, 32, 64]:
            x = torch.randn(b, 4, 16, dtype=torch.float16, device=dev)
            out = runner(x, inputs[1], inputs[2])
            out = out[0] if isinstance(out, (list, tuple)) else out
            ref = LayerNorm().eval()(x, inputs[1], inputs[2])
            self.assertEqual(out.shape, ref.shape)
            torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_layer_norm_backward_dynamic_batch(self):
        # Inductor decomposes native_layer_norm_backward into reduction and
        # pointwise Triton kernels. AOTI should launch the generated MUSA
        # Triton kernels without using a direct C-shim or ProxyExecutor. This
        # also covers three tensor outputs, SymIntArrayRef normalized shape,
        # and the fixed-size boolean output mask.
        class LayerNormBackward(torch.nn.Module):
            def forward(self, grad_out, x, mean, rstd, weight, bias):
                return torch.ops.aten.native_layer_norm_backward.default(
                    grad_out,
                    x,
                    (16,),
                    mean,
                    rstd,
                    weight,
                    bias,
                    (True, True, True),
                )

        dev = "musa"
        batch = Dim("batch", min=1, max=64)
        weight = torch.randn(16, dtype=torch.float16, device=dev)
        bias = torch.randn(16, dtype=torch.float16, device=dev)

        def make_inputs(b):
            x = torch.randn(b, 4, 16, dtype=torch.float16, device=dev)
            grad_out = torch.randn_like(x)
            _, mean, rstd = torch.ops.aten.native_layer_norm.default(
                x, (16,), weight, bias, 1e-5
            )
            return grad_out, x, mean, rstd, weight, bias

        inputs = make_inputs(8)
        ep = torch.export.export(
            LayerNormBackward().eval(),
            inputs,
            dynamic_shapes=(
                {0: batch},
                {0: batch},
                {0: batch},
                {0: batch},
                {},
                {},
            ),
            strict=False,
        )
        with patch("torch._inductor.ir.log.warning") as warning:
            pkg = torch._inductor.aoti_compile_and_package(ep)
        proxy_fallback_warnings = [
            call
            for call in warning.call_args_list
            if call.args
            and "native_layer_norm_backward" in str(call.args)
            and "missing a c-shim implementation" in str(call.args[0])
        ]
        self.assertFalse(proxy_fallback_warnings)
        runner = torch._inductor.aoti_load_package(pkg)

        for b in [1, 8, 16, 32, 64]:
            run_inputs = make_inputs(b)
            out = runner(*run_inputs)
            ref = LayerNormBackward().eval()(*run_inputs)
            self.assertEqual(len(out), len(ref))
            for actual, expected in zip(out, ref):
                self.assertEqual(actual.shape, expected.shape)
                torch.testing.assert_close(
                    actual.float(), expected.float(), atol=2e-2, rtol=2e-2
                )

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_pointwise_dynamic_batch(self):
        # Inductor fuses the pointwise add and relu operations into a Triton
        # kernel. AOTI should launch the generated MUSA Triton kernel rather
        # than route this graph through an ATen fallback C-shim.
        class Pointwise(torch.nn.Module):
            def forward(self, x, y):
                return torch.nn.functional.relu(x + y)

        dev = "musa"
        batch = Dim("batch", min=1, max=64)
        inputs = (
            torch.randn(8, 4, 16, dtype=torch.float16, device=dev),
            torch.randn(8, 4, 16, dtype=torch.float16, device=dev),
        )
        ep = torch.export.export(
            Pointwise().eval(),
            inputs,
            dynamic_shapes=({0: batch}, {0: batch}),
            strict=False,
        )
        pkg = torch._inductor.aoti_compile_and_package(ep)
        runner = torch._inductor.aoti_load_package(pkg)

        for b in [1, 8, 16, 32, 64]:
            x = torch.randn(b, 4, 16, dtype=torch.float16, device=dev)
            y = torch.randn(b, 4, 16, dtype=torch.float16, device=dev)
            out = runner(x, y)
            out = out[0] if isinstance(out, (list, tuple)) else out
            ref = Pointwise().eval()(x, y)
            self.assertEqual(out.shape, ref.shape)
            torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_adaptive_avg_pool_dynamic_batch(self):
        # Inductor keeps adaptive average pooling as an ATen fallback node.
        # This verifies that the SymIntArrayRef/list output-size argument is
        # passed through the direct MUSA C-shim, without using ProxyExecutor.
        class AdaptiveAvgPool(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))

        dev = "musa"
        batch = Dim("batch", min=1, max=64)
        inputs = (torch.randn(8, 3, 16, 16, dtype=torch.float16, device=dev),)
        ep = torch.export.export(
            AdaptiveAvgPool().eval(),
            inputs,
            dynamic_shapes=({0: batch},),
            strict=False,
        )
        with patch("torch._inductor.ir.log.warning") as warning:
            pkg = torch._inductor.aoti_compile_and_package(ep)
        proxy_fallback_warnings = [
            call
            for call in warning.call_args_list
            if call.args and "missing a c-shim implementation" in str(call.args[0])
        ]
        self.assertFalse(proxy_fallback_warnings)
        runner = torch._inductor.aoti_load_package(pkg)

        for b in [1, 8, 16, 32, 64]:
            x = torch.randn(b, 3, 16, 16, dtype=torch.float16, device=dev)
            out = runner(x)
            out = out[0] if isinstance(out, (list, tuple)) else out
            ref = AdaptiveAvgPool().eval()(x)
            self.assertEqual(out.shape, ref.shape)
            torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_sdpa_dynamic_batch_headdim(self):
        # Inductor lowers SDPA to a MUSA-specific attention implementation.
        # AOTI currently executes this low-level operator through ProxyExecutor
        # because it does not have a direct MUSA C-shim implementation. This
        # path is intentionally tested separately from convolution and pooling.
        class Attn(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        dev = "musa"
        mk = lambda b, head_dim: torch.randn(
            b, 4, 4, head_dim, dtype=torch.float16, device=dev
        )
        batch = Dim("batch", min=1, max=4096)
        qk_head_dim = Dim("qk_head_dim", min=16, max=128)
        v_head_dim = Dim("v_head_dim", min=16, max=128)
        ep = torch.export.export(
            Attn().eval(),
            (
                mk(8, 16),
                mk(8, 16),
                mk(8, 32),
            ),  # example batch = 8, qk_head_dim = 16, v_head_dim = 32
            dynamic_shapes=(
                {0: batch, 3: qk_head_dim},
                {0: batch, 3: qk_head_dim},
                {0: batch, 3: v_head_dim},
            ),
            strict=False,
        )
        pkg = torch._inductor.aoti_compile_and_package(ep)
        runner = torch._inductor.aoti_load_package(pkg)

        for b in [1, 5, 8, 16, 32, 64, 256, 1024, 4096]:
            for head_dim in [16, 64, 128]:
                q, k, v = mk(b, head_dim), mk(b, head_dim), mk(b, head_dim)
                out = runner(q, k, v)
                out = out[0] if isinstance(out, (list, tuple)) else out
                ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                self.assertEqual(out.shape, ref.shape)
                torch.testing.assert_close(
                    out.float(), ref.float(), atol=2e-2, rtol=2e-2
                )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if torch.musa.is_available():
        run_tests(needs="filelock")
