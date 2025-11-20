"""Test inductor package"""

# Owner(s): ["module: inductor"]
import copy
import functools
import unittest
import pytest

import torch
from torch._inductor import config
from torch._inductor.test_case import TestCase
from torch.testing._internal import common_utils

from torch_musa._inductor.package import load_package
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


def compile(model, example_inputs, dynamic_shapes, options, device):
    ep = torch.export.export(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    gm = ep.module()
    package_path = torch._inductor.aot_compile(gm, example_inputs, options=options)
    compiled_model = load_package(package_path, device)
    return compiled_model


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
            "aot_inductor.package": True,
            # TODO: "aot_inductor.force_mmap_weights": True,
        }
    ):
        torch.manual_seed(0)
        model = model.to(self.device)
        ref_model = copy.deepcopy(model)
        ref_inputs = copy.deepcopy(example_inputs)
        expected = ref_model(*ref_inputs)

        torch.manual_seed(0)
        compiled_model = compile(
            model,
            example_inputs,
            dynamic_shapes,
            options,
            self.device,
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


class AOTInductorTestPackagedABICompatibleMusa(TestCase):
    device = "musa"
    check_model = check_model


copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestPackagedABICompatibleMusa,
    "packaged_abi_compatible_musa",
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # cpp_extension N/A in fbcode
    if torch.musa.is_available():
        run_tests(needs="filelock")
