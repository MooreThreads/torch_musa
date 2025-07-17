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

import torch
import torch._export
import torch._inductor
from torch import nn
from torch._dynamo.testing import rand_strided, same
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import TestCase
from torch.export import Dim, export
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
            "abi_compatible": self.abi_compatible,
            "allow_stack_allocation": self.allow_stack_allocation,
            "use_minimal_arrayref_interface": self.use_minimal_arrayref_interface,
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
            "abi_compatible": self.abi_compatible,
            "allow_stack_allocation": self.allow_stack_allocation,
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
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x, y):
                return x + self.linear(y)

        example_inputs = (
            torch.randn(10, 10, device=self.device),
            torch.randn(10, 10, device=self.device),
        )
        self.check_model(Model(), example_inputs)

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
    def test_conv_freezing(self):
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

            with config.patch({"freezing": True}):
                self.check_model(Model(self.device), example_inputs)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_seq(self):
        layernorm = torch.nn.LayerNorm(10)
        net = torch.nn.Sequential(
            layernorm,
            torch.nn.ReLU(),
            layernorm,
            torch.nn.ReLU(),
        )

        example_inputs = (torch.randn(10, device=self.device),)
        self.check_model(net.eval(), example_inputs)

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

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_addmm_multiple_dynamic(self):
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
        dim0_a = Dim("dim0_a", min=1, max=2048)
        dynamic_shapes = {"a": {0: dim0_a}}
        list_example_inputs = [(a,)]
        batch = 2048
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        batch = 128
        list_example_inputs.append(
            (torch.randn(batch, M, K, device=self.device),),
        )
        self.check_model_with_multiple_inputs(
            model,
            list_example_inputs,
            dynamic_shapes=dynamic_shapes,
            options={
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

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

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_consecutive_compiles(self):
        """Test that compilation behaves correctly with cache hits"""

        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x + 1

        mod = TestModule()
        inp = torch.rand(1)
        mod(inp)
        mod2 = torch.fx.symbolic_trace(mod, concrete_args=[inp])
        so = torch._export.aot_compile(mod2, (inp,))
        assert so is not None
        # compile the 2nd time with cache hit
        so = torch._export.aot_compile(mod2, (inp,))
        assert so is not None

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_empty_graph(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x

        example_inputs = (torch.randn(8, 4, 4, device=self.device),)
        self.check_model(Model(), example_inputs)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_convolution(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, w, b):
                return torch.ops.aten.convolution(x, w, b, [4], [0], [1], True, [0], 1)

        example_inputs = (
            torch.randn([2, 32, 90], device=self.device),
            torch.randn([32, 16, 8], device=self.device),
            torch.randn([16], device=self.device),
        )
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            self.check_model(Model(), example_inputs)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_fqn(self):
        class NestedChild(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nestedchild3buffer = torch.nn.Buffer(torch.ones(2, 3) * 3)

            def forward(self, x):
                return x / self.nestedchild3buffer

        class Child1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                x = self.nested(x)
                return x + self.child1param

        class Child2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.child2buffer = torch.nn.Buffer(torch.ones(2, 3) * 2)

            def forward(self, x):
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3) * 4)
                )

            def forward(self, x):
                x = x * self.rootparam
                x = self.foo(x)
                x = self.bar(x)
                return x

        orig_eager = MyModule()

        self.check_model(MyModule(), (torch.randn(2, 3, device=self.device),))

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


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if torch.musa.is_available():
        run_tests(needs="filelock")
