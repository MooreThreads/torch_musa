"""Test inductor package"""

# Owner(s): ["module: inductor"]
import copy
import functools
import unittest
import pytest
import tempfile

import torch
from torch._inductor import config
from torch._inductor.package import load_package
from torch._inductor.test_case import TestCase
from torch.testing._internal import common_utils

# from torch_musa._inductor.package import load_package
import torch_musa
from torch_musa.testing.base_test_tool import _HAS_TRITON

# pylint: disable-all


def copy_tests(
    my_cls, other_cls, suffix, test_failures=None, xfail_prop=None
):  # noqa: B902
    for name, value in my_cls.__dict__.items():
        if name.startswith("test_"):

            @functools.wraps(value)
            def new_test(self, value=value):
                return value(self)

            # Copy __dict__ which may contain test metadata
            new_test.__dict__ = copy.deepcopy(value.__dict__)

            if xfail_prop is not None and hasattr(value, xfail_prop):
                new_test = unittest.expectedFailure(new_test)

            tf = test_failures and test_failures.get(name)
            if tf is not None and suffix in tf.suffixes:
                skip_func = (
                    unittest.skip("Skipped!")
                    if tf.is_skip
                    else unittest.expectedFailure
                )
                new_test = skip_func(new_test)

            setattr(other_cls, f"{name}_{suffix}", new_test)


def compile(
    model, example_inputs, dynamic_shapes, inductor_configs=None, package_path=None
):
    ep = torch.export.export(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    package_path = torch._inductor.aoti_compile_and_package(
        ep, package_path=package_path, inductor_configs=inductor_configs
    )  # type: ignore[arg-type]
    loaded = load_package(package_path)
    return loaded


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
    inductor_configs = {
        "aot_inductor.package": True,
        "aot_inductor.package_cpp_only": getattr(self, "package_cpp_only", True),
        "aot_inductor.force_mmap_weights": getattr(self, "force_mmap_weights", True),
    }
    with torch.no_grad():
        torch.manual_seed(0)
        model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(example_inputs)
        expected = ref_model(*ref_inputs)

        torch.manual_seed(0)
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            compiled_model = compile(
                model,
                example_inputs,
                dynamic_shapes,
                inductor_configs=inductor_configs,
                package_path=f.name,
            )

        actual = compiled_model(*example_inputs)

    self.assertEqual(actual, expected, atol=atol, rtol=rtol)


class AOTInductorTestsTemplate:

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_add(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        example_inputs = (
            torch.randn(16, 16, device=self.device),
            torch.randn(16, 16, device=self.device),
        )
        self.check_model(Model(), example_inputs)

    @pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
    def test_linear(self):
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
        self.check_model(Model(), example_inputs)


common_utils.instantiate_parametrized_tests(AOTInductorTestsTemplate)


class AOTInductorTestPackagedCppOnlyMmapMusa(TestCase):
    device = "musa"
    check_model = check_model
    package_cpp_only = True
    force_mmap_weights = True


class AOTInductorTestPackagedCppOnlyNoMmapMusa(TestCase):
    device = "musa"
    check_model = check_model
    package_cpp_only = True
    force_mmap_weights = False


class AOTInductorTestPackagedNoCppMmapMusa(TestCase):
    device = "musa"
    check_model = check_model
    package_cpp_only = False
    force_mmap_weights = True


class AOTInductorTestPackagedNoCppNoMmapMusa(TestCase):
    device = "musa"
    check_model = check_model
    package_cpp_only = False
    force_mmap_weights = False


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestPackagedCppOnlyMmapMusa,
    "cpp_only_mmap_musa",
)
copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestPackagedCppOnlyNoMmapMusa,
    "cpp_only_no_mmap_musa",
)
copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestPackagedNoCppMmapMusa,
    "no_cpp_mmap_musa",
)
copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestPackagedNoCppNoMmapMusa,
    "no_cpp_no_mmap_musa",
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if torch.musa.is_available():
        run_tests(needs="filelock")
