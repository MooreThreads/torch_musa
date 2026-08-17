# Owner(s): ["module: cpp"]

import gc
import math
import os
import sys
import sysconfig
import unittest
from pathlib import Path

import torch
import torch_musa
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    onlyPRIVATEUSE1,
)
from torch.testing._internal.common_dtype import all_types_and
from torch.testing._internal.common_utils import (
    install_cpp_extension,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    skipIfWindows,
    TestCase,
    xfailIfTorchDynamo,
)
from torch_musa.utils.musa_extension import MUSA_HOME, include_paths

def _torchVersionLessThan(major, minor):
    version_parts = torch.__version__.split(".")
    current_major = int(version_parts[0])
    current_minor = int(
        version_parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
    )
    return (current_major < major) or (current_major == major and current_minor < minor)


def skipIfTorchVersionLessThan(major, minor):
    """Skip test if PyTorch version is less than specified version."""

    def decorator(func):
        reason = f"Test requires PyTorch >= {major}.{minor}, current version is {torch.__version__}"
        return unittest.skipIf(_torchVersionLessThan(major, minor), reason)(func)

    return decorator


def _is_musa_device(device):
    return torch.device(device).type == "musa"


def _assert_tensor_metadata_equal(testcase, actual, expected, *, exact_device=False, exact_layout=False):
    testcase.assertEqual(actual.shape, expected.shape)
    testcase.assertEqual(actual.dtype, expected.dtype)
    if exact_device:
        testcase.assertEqual(actual.device, expected.device)
    if exact_layout:
        testcase.assertEqual(actual.layout, expected.layout)


def _has_extension(extension_root):
    install_root = extension_root / "install"
    if not install_root.exists():
        return False

    mod_install_dir = None
    for root, directories, _ in os.walk(install_root):
        for directory in directories:
            if "-packages" in directory:
                mod_install_dir = os.path.join(root, directory)

    if mod_install_dir is None:
        return False

    sys.path.insert(0, mod_install_dir)
    return True

class TestLibtorchMusa(TestCase):
    """
    Tests for versioned libtorch_agnostic extensions.

    This test class supports testing:

    - libtorch_agn_2_9: Extension built with TORCH_TARGET_VERSION=2.9.0
    - libtorch_agn_2_10: Extension built with TORCH_TARGET_VERSION=2.10.0
    - libtorch_agn_2_11: Extension built with TORCH_TARGET_VERSION=2.11.0

    Tests should be decorated with @skipIfTorchVersionLessThan to indicate the
    version that they target.
    """

    @classmethod
    def setUpClass(cls):
        # Build versioned extensions
        base_dir = Path(__file__).parent
        extension_2_9_root = base_dir / "libtorch_musa_2_9_extension"
        extension_2_10_root = base_dir / "libtorch_musa_2_10_extension"
        extension_2_11_root = base_dir / "libtorch_musa_2_11_extension"

        if _has_extension(extension_2_9_root):
            import libtorch_agn_2_9  # noqa: F401
        else:
            install_cpp_extension(extension_root=extension_2_9_root)

        if _has_extension(extension_2_10_root):
            import libtorch_agn_2_10  # noqa: F401
        else:
            install_cpp_extension(extension_root=extension_2_10_root)

        if _has_extension(extension_2_11_root):
            import libtorch_agn_2_11  # noqa: F401
        else:
            install_cpp_extension(extension_root=extension_2_11_root)

    @onlyCPU
    def test_slow_sgd(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        param = torch.rand(5, device=device)
        grad = torch.rand_like(param)
        weight_decay = 0.01
        lr = 0.001
        maximize = False

        new_param = libtorch_musa.ops.sgd_out_of_place(
            param, grad, weight_decay, lr, maximize
        )
        torch._fused_sgd_(
            (param,),
            (grad,),
            (),
            weight_decay=weight_decay,
            momentum=0.0,
            lr=lr,
            dampening=0.0,
            nesterov=False,
            maximize=maximize,
            is_first_step=False,
        )
        self.assertEqual(new_param, param)

    @onlyPRIVATEUSE1
    def test_identity_does_not_hog_memory(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        def _run_identity(prior_mem):
            t = torch.rand(32, 32, device=device)
            self.assertGreater(torch.musa.memory_allocated(device), prior_mem)
            identi_t = libtorch_musa.ops.identity(t)
            if identi_t is not t:
                raise AssertionError("Expected identity op to return the same tensor")

        init_mem = torch.musa.memory_allocated(device)

        for _ in range(3):
            _run_identity(init_mem)
            curr_mem = torch.musa.memory_allocated(device)
            self.assertEqual(curr_mem, init_mem)

    def test_exp_neg_is_leaf(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t1 = torch.rand(2, 3, device=device)
        t2 = torch.rand(3, 2, device=device)
        t3 = torch.rand(2, device=device)

        exp, neg, is_leaf = libtorch_musa.ops.exp_neg_is_leaf(t1, t2, t3)
        self.assertEqual(exp, torch.exp(t1))
        self.assertEqual(neg, torch.neg(t2))
        self.assertEqual(is_leaf, t3.is_leaf)

    def test_my_abs(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(32, 16, device=device) - 0.5
        res = libtorch_musa.ops.my_abs(t)
        self.assertEqual(res, torch.abs(t))

        def _make_musa_tensors(prior_mem):
            musa_t = libtorch_musa.ops.my_abs(t)
            self.assertGreater(torch.musa.memory_allocated(device), prior_mem)
            self.assertEqual(musa_t, torch.abs(t))

        if t.is_musa:
            init_mem = torch.musa.memory_allocated(device)
            for _ in range(3):
                _make_musa_tensors(init_mem)
                curr_mem = torch.musa.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_neg_exp(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(32, 16, device=device) - 0.5
        res = libtorch_musa.ops.neg_exp(t)
        self.assertEqual(res, torch.neg(torch.exp(t)))

        def _make_musa_tensors(prior_mem):
            musa_res = libtorch_musa.ops.neg_exp(t)
            self.assertGreater(torch.musa.memory_allocated(device), prior_mem)
            self.assertEqual(musa_res, torch.neg(torch.exp(t)))

        if t.is_musa:
            init_mem = torch.musa.memory_allocated(device)
            for _ in range(3):
                _make_musa_tensors(init_mem)
                curr_mem = torch.musa.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_divide_neg_exp(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.zeros(2, 3, device=device) - 0.5
        res = libtorch_musa.ops.divide_neg_exp(t)
        self.assertEqual(res, torch.neg(t) / torch.exp(t))

        def _make_musa_tensors(prior_mem):
            musa_res = libtorch_musa.ops.divide_neg_exp(t)
            self.assertGreater(torch.musa.memory_allocated(device), prior_mem)
            self.assertEqual(musa_res, torch.neg(t) / torch.exp(t))

        if t.is_musa:
            init_mem = torch.musa.memory_allocated(device)
            for _ in range(3):
                _make_musa_tensors(init_mem)
                curr_mem = torch.musa.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    def test_is_contiguous(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(2, 7, device=device)
        self.assertTrue(libtorch_musa.ops.is_contiguous(t))
        self.assertFalse(libtorch_musa.ops.is_contiguous(t.transpose(0, 1)))

    def test_my_ones_like(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(3, 1, device=device) - 0.5
        cpu_t = libtorch_musa.ops.my_ones_like(t, "cpu")
        self.assertEqual(cpu_t, torch.ones_like(t, device="cpu"))

        def _make_musa_tensors(prior_mem):
            musa_t = libtorch_musa.ops.my_ones_like(t, device)
            self.assertGreater(torch.musa.memory_allocated(device), prior_mem)
            self.assertEqual(musa_t, torch.ones_like(t, device=device))

        if t.is_musa:
            init_mem = torch.musa.memory_allocated(device)
            for _ in range(3):
                _make_musa_tensors(init_mem)
                curr_mem = torch.musa.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    # @skipIfTorchVersionLessThan(2, 11)  # Requires 2.11 for Float8_e8m0fnu support
    # def test_my_ones_like_with_Float8_e8m0fnu(self, device):
    #     import libtorch_musa_2_11 as libtorch_musa

    #     if _is_musa_device(device):
    #         self.skipTest("MUSA does not support Float8_e8m0fnu tensors yet")

    #     t = torch.zeros(3, 1, device=device, dtype=torch.float8_e8m0fnu)
    #     cpu_t = libtorch_musa.ops.my_ones_like(t, "cpu")
    #     self.assertEqual(cpu_t, torch.ones_like(t, device="cpu"))

    #     def _make_musa_tensors(prior_mem):
    #         musa_t = libtorch_musa.ops.my_ones_like(t, device)
    #         self.assertGreater(torch.musa.memory_allocated(device), prior_mem)
    #         self.assertEqual(musa_t, torch.ones_like(t, device=device))

    #     if t.is_musa:
    #         init_mem = torch.musa.memory_allocated(device)
    #         for _ in range(3):
    #             _make_musa_tensors(init_mem)
    #             curr_mem = torch.musa.memory_allocated(device)
    #             self.assertEqual(curr_mem, init_mem)

    def test_my_transpose(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(2, 7, device=device)
        out = libtorch_musa.ops.my_transpose(t, 0, 1)
        self.assertEqual(out, torch.transpose(t, 0, 1))

        with self.assertRaisesRegex(RuntimeError, "API call failed"):
            libtorch_musa.ops.my_transpose(t, 1, 2)

    def test_my_empty_like(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            # set use_deterministic_algorithms to fill uninitialized memory
            torch.use_deterministic_algorithms(True)

            t = torch.rand(2, 7, device=device)
            out = libtorch_musa.ops.my_empty_like(t)
            self.assertTrue(id(out != id(t)))
            expected = torch.empty_like(t)
            if _is_musa_device(device):
                _assert_tensor_metadata_equal(self, out, expected, exact_device=True)
            else:
                self.assertEqual(out, expected)
        finally:
            torch.use_deterministic_algorithms(deterministic)

    @onlyCPU
    def test_my_zero_(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(2, 7, device=device)
        out = libtorch_musa.ops.my_zero_(t)
        self.assertEqual(id(out), id(t))
        self.assertEqual(out, torch.zeros_like(t))

    def test_my_amax(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(2, 7, device=device)
        out = libtorch_musa.ops.my_amax(t)
        self.assertEqual(out, torch.amax(t, 0))

    def test_my_amax_vec(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(2, 7, 5, device=device)
        out = libtorch_musa.ops.my_amax_vec(t)
        self.assertEqual(out, torch.amax(t, (0, 1)))

    def test_my_is_cpu(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(2, 7, device=device)
        out = libtorch_musa.ops.my_is_cpu(t)
        self.assertEqual(out, t.is_cpu)

    def test_fill_infinity(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(3, 4, device=device)
        out = libtorch_musa.ops.fill_infinity(t)

        self.assertEqual(id(out), id(t))
        expected = torch.full_like(t, math.inf)
        self.assertEqual(out, expected)

    @onlyCPU
    def test_default_constructor(self):
        import libtorch_agn_2_9 as libtorch_musa

        defined_tensor_is_defined = libtorch_musa.ops.test_default_constructor(True)
        self.assertTrue(defined_tensor_is_defined)

        undefined_tensor_is_defined = libtorch_musa.ops.test_default_constructor(
            False
        )
        self.assertFalse(undefined_tensor_is_defined)

    def test_my_pad(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.rand(2, 3, device=device)
        out = libtorch_musa.ops.my_pad(t)
        expected = torch.nn.functional.pad(t, [1, 2, 2, 1], "constant", 0.0)
        self.assertEqual(out, expected)

    def test_my_narrow(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.randn(2, 5, device=device)

        dim0 = 0
        start0 = 0
        length0 = 1
        out0 = libtorch_musa.ops.my_narrow(t, dim0, start0, length0)
        expected0 = torch.narrow(t, dim0, start0, length0)
        self.assertEqual(out0, expected0)

    @onlyCPU
    def test_my_narrow_symint(self, device):
        op = torch.ops.libtorch_agn_2_9.my_narrow_symint

        t1 = torch.randn(4, 5, device=device)
        out = op(t1, 0, 1, 2)
        expected = torch.narrow(t1, 0, 1, 2)
        self.assertEqual(out, expected)

        torch.library.opcheck(
            torch.ops.libtorch_agn_2_9.my_narrow_symint,
            (t1, 0, 1, 2),
        )

        # Below is a real example to confirm recompilation doesn't happen
        # Wrap in a function so shape computation happens inside the compiled
        # region — t.shape[0] becomes a symbolic SymInt during tracing
        def fn(t):
            return op(t, 0, 2, t.shape[0] - 2)

        cnt = CompileCounter()
        compiled_fn = torch.compile(fn, dynamic=True, backend=cnt)

        out2 = compiled_fn(t1)
        self.assertEqual(out2, torch.narrow(t1, 0, 2, t1.shape[0] - 2))
        frame_count = cnt.frame_count

        # Second call with different shape should not recompile
        t2 = torch.randn(6, 3, device=device)
        out3 = compiled_fn(t2)
        self.assertEqual(out3, torch.narrow(t2, 0, 2, t2.shape[0] - 2))
        self.assertEqual(cnt.frame_count, frame_count)

    @onlyPRIVATEUSE1
    @deviceCountAtLeast(2)
    def test_device_guard(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        device_index = 1
        out = libtorch_musa.ops.test_device_guard(device_index)
        self.assertEqual(out, device_index)

    @onlyPRIVATEUSE1
    @deviceCountAtLeast(2)
    def test_device_guard_set_index(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        # This test creates a DeviceGuard with index 1, then sets it to index 0
        # and returns the current device (should be 0)
        out = libtorch_musa.ops.test_device_guard_set_index()
        self.assertEqual(out, 0)

    @onlyPRIVATEUSE1
    def test_stream(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        stream = torch.musa.Stream()
        device = torch.musa.current_device()

        with stream:
            expected_stream_id = torch.musa.current_stream(0).stream_id
            stream_id = libtorch_musa.ops.test_stream(device)

        self.assertEqual(stream_id, expected_stream_id)

    @onlyPRIVATEUSE1
    @deviceCountAtLeast(2)
    def test_get_current_device_index(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        prev_device = torch.musa.current_device()

        try:
            expected_device = 1
            torch.musa.set_device(expected_device)

            current_device = libtorch_musa.ops.test_get_current_device_index()
            self.assertEqual(current_device, expected_device)
        finally:
            torch.musa.set_device(prev_device)

    def test_my_new_empty_dtype_variant(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            # set use_deterministic_algorithms to fill uninitialized memory
            torch.use_deterministic_algorithms(True)
            t = torch.randn(3, 4, device=device)
            out = libtorch_musa.ops.my_new_empty_dtype_variant(t)
            ref_out = t.new_empty((2, 5), dtype=torch.bfloat16)

            if _is_musa_device(device):
                _assert_tensor_metadata_equal(self, out, ref_out, exact_device=True)
            else:
                self.assertEqual(out, ref_out, exact_device=True)
        finally:
            torch.use_deterministic_algorithms(deterministic)

    def test_my_new_zeros_dtype_variant(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.randn(3, 4, device=device)
        out = libtorch_musa.ops.my_new_zeros_dtype_variant(t)
        ref_out = t.new_zeros((2, 5), dtype=torch.float)
        self.assertEqual(out, ref_out, exact_device=True)

    def test_my_copy_(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        dst = torch.empty(2, 5, device=device)
        src = torch.randn(2, 5, device=device)

        result = libtorch_musa.ops.my_copy_(dst, src, False)
        expected = src
        self.assertEqual(result, expected)
        self.assertEqual(result.data_ptr(), dst.data_ptr())

    def test_my_clone(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.randn(2, 5, device=device)

        result = libtorch_musa.ops.my_clone(t)
        expected = t.clone()
        self.assertEqual(result, expected)
        self.assertNotEqual(result.data_ptr(), expected.data_ptr())
        self.assertEqual(result.stride(), expected.stride())

    @skipIfTorchVersionLessThan(2, 10)
    def test_my__foreach_mul_(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        N = 5
        tensors = [torch.rand(32, 16, device=device) for _ in range(N)]
        tensors_c = [t.clone() for t in tensors]
        others = [torch.rand(32, 16, device=device) for _ in range(N)]

        libtorch_musa.ops.my__foreach_mul_(tensors, others)
        expected_values = torch._foreach_mul(tensors_c, others)

        for tensor_t, expected_t in zip(tensors, expected_values):
            self.assertEqual(tensor_t, expected_t)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my__foreach_mul(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        N = 5
        tensors = [torch.rand(32, 16, device=device) for _ in range(N)]
        others = [torch.rand(32, 16, device=device) for _ in range(N)]

        result = libtorch_musa.ops.my__foreach_mul(tensors, others)
        expected = torch._foreach_mul(tensors, others)

        for result_t, expected_t in zip(result, expected):
            self.assertEqual(result_t, expected_t)

        def _make_musa_tensors(prior_mem):
            musa_res = libtorch_musa.ops.my__foreach_mul(tensors, others)
            self.assertGreater(torch.musa.memory_allocated(device), prior_mem)

            expected = torch._foreach_mul(tensors, others)
            for result_t, expected_t in zip(musa_res, expected):
                self.assertEqual(result_t, expected_t)

        if tensors[0].is_musa:
            init_mem = torch.musa.memory_allocated(device)
            for _ in range(3):
                _make_musa_tensors(init_mem)
                curr_mem = torch.musa.memory_allocated(device)
                self.assertEqual(curr_mem, init_mem)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_make_tensor_clones_and_call_foreach(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        t1 = torch.rand(2, 5, device=device)
        t2 = torch.rand(3, 4, device=device)
        result = libtorch_musa.ops.make_tensor_clones_and_call_foreach(t1, t2)
        self.assertEqual(result[0], t1 * t1)
        self.assertEqual(result[1], t2 * t2)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_device(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        musa_device = libtorch_musa.ops.test_device_constructor(
            is_musa=True, index=1, use_str=False
        )
        self.assertEqual(musa_device, torch.device("musa:1"))
        musa_device = libtorch_musa.ops.test_device_constructor(
            is_musa=True, index=1, use_str=True
        )
        self.assertEqual(musa_device, torch.device("musa:1"))

        self.assertEqual(libtorch_musa.ops.test_device_index(musa_device), 1)
        self.assertTrue(
            libtorch_musa.ops.test_device_equality(
                musa_device, torch.device("musa:1")
            )
        )
        self.assertFalse(
            libtorch_musa.ops.test_device_equality(
                musa_device, torch.device("musa:0")
            )
        )
        self.assertFalse(libtorch_musa.ops.test_device_is_cpu(musa_device))
        self.assertTrue(libtorch_musa.ops.test_device_is_musa(musa_device))

        musa_0_device = libtorch_musa.ops.test_device_set_index(musa_device, 0)
        self.assertEqual(musa_0_device, torch.device("musa:0"))

        cpu_device = libtorch_musa.ops.test_device_constructor(False, 0, False)
        self.assertEqual(cpu_device, torch.device("cpu"))
        self.assertTrue(
            libtorch_musa.ops.test_device_equality(cpu_device, torch.device("cpu"))
        )
        self.assertTrue(libtorch_musa.ops.test_device_is_cpu(cpu_device))
        self.assertFalse(libtorch_musa.ops.test_device_is_musa(cpu_device))
        self.assertFalse(
            libtorch_musa.ops.test_device_equality(cpu_device, musa_device)
        )

        with self.assertRaisesRegex(
            RuntimeError, "Device index 129 is out of range for int8_t"
        ):
            libtorch_musa.ops.test_device_constructor(
                is_musa=True, index=129, use_str=False
            )

        with self.assertRaisesRegex(
            RuntimeError, "Device index 129 is out of range for int8_t"
        ):
            libtorch_musa.ops.test_device_set_index(musa_device, 129)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    @deviceCountAtLeast(2)
    def test_tensor_device(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(2, 3)
        self.assertEqual(libtorch_musa.ops.test_tensor_device(t), t.device)

        t_musa = torch.randn(2, 3, device="musa")
        self.assertEqual(
            libtorch_musa.ops.test_tensor_device(t_musa), t_musa.device
        )

        t_musa_1 = torch.randn(2, 3, device="musa:1")
        self.assertEqual(
            libtorch_musa.ops.test_tensor_device(t_musa_1), t_musa_1.device
        )

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCPU
    def test_parallel_for(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        num_threads = torch.get_num_threads()
        size = 100
        grain_size = 10
        expected_num_threads_used = min(
            (size + grain_size - 1) // grain_size, num_threads
        )

        result = libtorch_musa.ops.test_parallel_for(size, grain_size)
        result_thread_ids = torch.unique(torch.bitwise_right_shift(result, 32))
        result_values = torch.bitwise_and(result, 0xFFFFFFFF)
        expected = torch.arange(size, dtype=torch.int64)

        self.assertEqual(result_values, expected)
        self.assertEqual(result_thread_ids, torch.arange(expected_num_threads_used))

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCPU
    def test_get_num_threads(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        num_threads = libtorch_musa.ops.test_get_num_threads()
        expected_num_threads = torch.get_num_threads()
        self.assertEqual(num_threads, expected_num_threads)

    @skipIfTorchVersionLessThan(2, 10)
    @parametrize("layout", [None, torch.strided, torch.sparse_coo])
    @parametrize("memory_format", [None, torch.channels_last, torch.contiguous_format])
    def test_my_empty(self, device, layout, memory_format):
        import libtorch_agn_2_10 as libtorch_musa

        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            # set use_deterministic_algorithms to fill uninitialized memory
            torch.use_deterministic_algorithms(True)

            # Use 4D size for channels_last, 2D otherwise
            size = [2, 3, 4, 5] if memory_format == torch.channels_last else [2, 3]

            # sparse_coo layout doesn't support memory_format parameter
            if layout == torch.sparse_coo and memory_format is not None:
                return

            # Test default parameters
            result = libtorch_musa.ops.my_empty(
                size, None, layout, None, None, memory_format
            )
            expected = torch.empty(size, layout=layout, memory_format=memory_format)
            if _is_musa_device(device) and result.device.type == "musa":
                _assert_tensor_metadata_equal(
                    self, result, expected, exact_device=True, exact_layout=True
                )
            else:
                self.assertEqual(result, expected, exact_device=True, exact_layout=True)

            # Test with dtype
            result_float = libtorch_musa.ops.my_empty(
                size, torch.float32, layout, None, None, memory_format
            )
            expected_float = torch.empty(
                size,
                dtype=torch.float32,
                layout=layout,
                memory_format=memory_format,
            )
            if _is_musa_device(device) and result_float.device.type == "musa":
                _assert_tensor_metadata_equal(
                    self,
                    result_float,
                    expected_float,
                    exact_device=True,
                    exact_layout=True,
                )
            else:
                self.assertEqual(
                    result_float, expected_float, exact_device=True, exact_layout=True
                )

            # Test with dtype and device
            result_with_device = libtorch_musa.ops.my_empty(
                size, torch.float64, layout, device, None, memory_format
            )
            expected_with_device = torch.empty(
                size,
                dtype=torch.float64,
                layout=layout,
                device=device,
                memory_format=memory_format,
            )
            if _is_musa_device(device):
                _assert_tensor_metadata_equal(
                    self,
                    result_with_device,
                    expected_with_device,
                    exact_device=True,
                    exact_layout=True,
                )
            else:
                self.assertEqual(
                    result_with_device,
                    expected_with_device,
                    exact_device=True,
                    exact_layout=True,
                )

            # Verify layout if specified
            if layout is not None:
                self.assertEqual(result_with_device.layout, layout)

            # Verify memory format if specified
            if memory_format == torch.channels_last:
                self.assertTrue(
                    result_with_device.is_contiguous(memory_format=torch.channels_last)
                )
            elif memory_format == torch.contiguous_format:
                self.assertTrue(result_with_device.is_contiguous())

            # Test pin_memory on MUSA (only once, not for every parameter combination)
            if device == "musa" and layout is None and memory_format is None:
                result_pinned = libtorch_musa.ops.my_empty(
                    [2, 3], torch.float32, None, "cpu", True, None
                )
                expected_pinned = torch.empty(
                    [2, 3], dtype=torch.float32, device="cpu", pin_memory=True
                )
                self.assertEqual(
                    result_pinned,
                    expected_pinned,
                    exact_device=True,
                    exact_layout=True,
                )
                self.assertTrue(result_pinned.is_pinned())
        finally:
            torch.use_deterministic_algorithms(deterministic)

    def test_my_flatten(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.randn(2, 3, 4, device=device)
        result = libtorch_musa.ops.my_flatten(t)
        expected = torch.flatten(t)
        self.assertEqual(result, expected)

        result_start = libtorch_musa.ops.my_flatten(t, 1)
        expected_start = torch.flatten(t, 1)
        self.assertEqual(result_start, expected_start)

        result_range = libtorch_musa.ops.my_flatten(t, 2, -1)
        expected_range = torch.flatten(t, 2, -1)
        self.assertEqual(result_range, expected_range)

    @onlyCPU
    def test_my_optional_tensor_ref(self, device):
        """Test TORCH_BOX with const std::optional<Tensor>& parameter."""
        import libtorch_agn_2_9 as libtorch_musa

        # Test with a tensor provided
        t = torch.randn(5, device=device)
        result = libtorch_musa.ops.my_optional_tensor_ref(t, 10)
        self.assertEqual(result, t)

        # Test with None (should return zeros tensor of specified size)
        result_none = libtorch_musa.ops.my_optional_tensor_ref(None, 7)
        expected_zeros = torch.zeros(7)
        self.assertEqual(result_none, expected_zeros)
        self.assertEqual(result_none.shape, (7,))

    def test_my_storage_offset(self, device):
        """Test storage_offset method on Tensor."""
        import libtorch_agn_2_9 as libtorch_musa

        # Test with a regular tensor (storage_offset should be 0)
        t = torch.randn(3, 4, device=device)
        result = libtorch_musa.ops.my_storage_offset(t)
        self.assertEqual(result, t.storage_offset())
        self.assertEqual(result, 0)

        # Test with a sliced tensor (storage_offset should be non-zero)
        t_sliced = t[1:]
        result_sliced = libtorch_musa.ops.my_storage_offset(t_sliced)
        self.assertEqual(result_sliced, t_sliced.storage_offset())
        self.assertEqual(result_sliced, 4)  # 1 row * 4 columns

        # Test with a view with offset
        t_view = t.view(-1)[2:]
        result_view = libtorch_musa.ops.my_storage_offset(t_view)
        self.assertEqual(result_view, t_view.storage_offset())
        self.assertEqual(result_view, 2)

    @dtypes(*all_types_and(torch.float16, torch.bool))
    def test_my_element_size(self, device, dtype):
        """Test element_size method on Tensor."""
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.zeros(2, 3, device=device, dtype=dtype)
        result = libtorch_musa.ops.my_element_size(t)
        self.assertEqual(result, t.element_size())

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_reshape(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(2, 3, 4, device=device)

        result = libtorch_musa.ops.my_reshape(t, [6, 4])
        expected = torch.reshape(t, [6, 4])
        self.assertEqual(result, expected)

        result_infer = libtorch_musa.ops.my_reshape(t, [-1, 4])
        expected_infer = torch.reshape(t, [-1, 4])
        self.assertEqual(result_infer, expected_infer)

        result_flat = libtorch_musa.ops.my_reshape(t, [-1])
        expected_flat = torch.reshape(t, [-1])
        self.assertEqual(result_flat, expected_flat)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_view(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(2, 3, 4, device=device)

        result = libtorch_musa.ops.my_view(t, [6, 4])
        expected = t.view([6, 4])
        self.assertEqual(result, expected)

        result_infer = libtorch_musa.ops.my_view(t, [-1, 4])
        expected_infer = t.view([-1, 4])
        self.assertEqual(result_infer, expected_infer)

        result_flat = libtorch_musa.ops.my_view(t, [-1])
        expected_flat = t.view([-1])
        self.assertEqual(result_flat, expected_flat)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_shape(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        expected = (3, 5)
        t = torch.rand(*expected, device=device)
        shape = libtorch_musa.ops.my_shape(t)
        self.assertEqual(shape, expected)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_sum(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(3, 4, 5, device=device)

        result = libtorch_musa.ops.my_sum(t, [0])
        expected = torch.sum(t, [0])
        self.assertEqual(result, expected)

        result_multi = libtorch_musa.ops.my_sum(t, [0, 2])
        expected_multi = torch.sum(t, [0, 2])
        self.assertEqual(result_multi, expected_multi)

        result_keepdim = libtorch_musa.ops.my_sum(t, [1], True)
        expected_keepdim = torch.sum(t, [1], keepdim=True)
        self.assertEqual(result_keepdim, expected_keepdim)

        result_dtype = libtorch_musa.ops.my_sum(t, [0], False, t.dtype)
        expected_dtype = torch.sum(t, [0], dtype=t.dtype)
        self.assertEqual(result_dtype, expected_dtype)

        # Test sum without dim (sum all elements)
        result_all = libtorch_musa.ops.my_sum(t)
        expected_all = torch.sum(t)
        self.assertEqual(result_all, expected_all)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_sum_out(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(3, 4, 5, device=device)

        out = torch.empty(4, 5, device=device)
        result = libtorch_musa.ops.my_sum_out(out, t, [0])
        expected = torch.sum(t, [0])
        self.assertEqual(out, expected)
        self.assertEqual(id(result), id(out))

        out_keepdim = torch.empty(3, 1, 5, device=device)
        libtorch_musa.ops.my_sum_out(out_keepdim, t, [1], True)
        expected_keepdim = torch.sum(t, [1], keepdim=True)
        self.assertEqual(out_keepdim, expected_keepdim)

        out_dtype = torch.empty(4, 5, dtype=t.dtype, device=device)
        libtorch_musa.ops.my_sum_out(out_dtype, t, [0], False, t.dtype)
        expected_dtype = torch.sum(t, [0], dtype=t.dtype)
        self.assertEqual(out_dtype, expected_dtype)

        out_all = torch.empty([], device=device)
        libtorch_musa.ops.my_sum_out(out_all, t)
        expected_all = torch.sum(t)
        self.assertEqual(out_all, expected_all)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_sum_all(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(3, 4, 5, device=device)

        # Test my_sum_all (sums all elements, returns scalar)
        result = libtorch_musa.ops.my_sum_all(t)
        expected = torch.sum(t)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([]))

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_sum_dim1(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(3, 4, 5, device=device)

        # Test my_sum_dim1 (sums along dimension 1)
        result = libtorch_musa.ops.my_sum_dim1(t)
        expected = torch.sum(t, dim=1)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([3, 5]))

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_full(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        # Test basic full with default parameters
        result = libtorch_musa.ops.my_full([2, 3], 3.14)
        expected = torch.full([2, 3], 3.14)
        self.assertEqual(result, expected)

        # Test with dtype
        result_dtype = libtorch_musa.ops.my_full([3, 4], 42.0, dtype=torch.int64)
        expected_dtype = torch.full([3, 4], 42, dtype=torch.int64)
        self.assertEqual(result_dtype, expected_dtype)

        # Test with device
        result_device = libtorch_musa.ops.my_full([2, 2], 1.5, device=device)
        expected_device = torch.full([2, 2], 1.5, device=device)
        self.assertEqual(result_device, expected_device, exact_device=True)

        # Test with dtype and device
        result_both = libtorch_musa.ops.my_full(
            [4, 5], 2.5, dtype=torch.float64, device=device
        )
        expected_both = torch.full([4, 5], 2.5, dtype=torch.float64, device=device)
        self.assertEqual(result_both, expected_both, exact_device=True)

    def test_mv_tensor_accessor(self, device):
        import libtorch_agn_2_9 as libtorch_musa

        m = torch.rand(3, 5, device=device)
        v = torch.rand(5, device=device)
        result = libtorch_musa.ops.mv_tensor_accessor(m, v)
        expected = torch.mv(m, v)
        self.assertEqual(result, expected)

        # non-contiguous inputs
        m = torch.rand(3 * 2, 5 * 3, device=device)[::2, ::3]
        v = torch.rand(5 * 4, device=device)[::4]
        result = libtorch_musa.ops.mv_tensor_accessor(m, v)
        expected = torch.mv(m, v)
        self.assertEqual(result, expected)

    @skipIfTorchVersionLessThan(2, 10)
    def test_get_any_data_ptr(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.empty(2, 5, device=device, dtype=torch.float32)
        expected_p = t.data_ptr()

        for mutable in [True, False]:
            p = libtorch_musa.ops.get_any_data_ptr(t, mutable)
            self.assertEqual(p, expected_p)

    def get_supported_dtypes_for_data_ptr(self):
        """Return a list of dtypes that are supported for casting Tensor data_ptrs.

        This is an intersection of supported dtypes in the stable ABI and the
        supported dtypes in TensorMethods.cpp/tensor_inl.h.
        """
        supported_by_2_10 = [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float64,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float8_e5m2fnuz,
            torch.float8_e4m3fnuz,
            torch.complex32,
            torch.complex64,
            torch.complex128,
            torch.bool,
        ]

        supported_by_2_11 = [
            # torch.float8_e8m0fnu,
        ]

        return supported_by_2_10 + (
            supported_by_2_11 if not _torchVersionLessThan(2, 11) else []
        )

    @skipIfTorchVersionLessThan(2, 10)
    def test_get_template_any_data_ptr(self, device):
        # if _torchVersionLessThan(2, 11):
        import libtorch_agn_2_10 as libtorch_musa
        # else:
            # import libtorch_agn_2_11 as libtorch_musa

        supported_dtypes = self.get_supported_dtypes_for_data_ptr()
        for dtype in supported_dtypes:
            t = torch.empty(2, 5, device=device, dtype=dtype)
            expected_p = t.data_ptr()

            for rdtype in supported_dtypes:
                if dtype == rdtype:
                    for mutable in [True, False]:
                        p = libtorch_musa.ops.get_template_any_data_ptr(
                            t, rdtype, mutable
                        )
                        self.assertEqual(p, expected_p)
                else:
                    for mutable in [True, False]:
                        with self.assertRaisesRegex(
                            RuntimeError, "expected scalar type.* but found"
                        ):
                            libtorch_musa.ops.get_template_any_data_ptr(
                                t, rdtype, mutable
                            )

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_my_get_curr_musa_blas_handle(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        res = libtorch_musa.ops.my_get_curr_musa_blas_handle()
        expected = torch.musa.current_blas_handle()
        self.assertEqual(res, expected)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my_string_op(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libtorch_musa.ops.my_string_op(t, "dim", "ice")
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "ice"])
        self.assertEqual(result_dim, t.dim())

        size_vec, result_size = libtorch_musa.ops.my_string_op(t, "size", "cream")
        self.assertEqual(size_vec, ["size", str(t.size(0)), "cream"])
        self.assertEqual(result_size, t.size(0))

        stride_vec, result_stride = libtorch_musa.ops.my_string_op(
            t, "stride", "cake"
        )
        self.assertEqual(stride_vec, ["stride", str(t.stride(0)), "cake"])
        self.assertEqual(result_stride, t.stride(0))

        with self.assertRaisesRegex(RuntimeError, "Unsupported accessor value: "):
            libtorch_musa.ops.my_string_op(t, "invalid", "")

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my__foreach_mul_vec(self, device):
        """Test my__foreach_mul_vec which uses const std::vector<Tensor>& parameters."""
        import libtorch_agn_2_10 as libtorch_musa

        N = 5
        tensors = [torch.rand(32, 16, device=device) for _ in range(N)]
        others = [torch.rand(32, 16, device=device) for _ in range(N)]

        result = libtorch_musa.ops.my__foreach_mul_vec(tensors, others)
        expected = torch._foreach_mul(tensors, others)

        for result_t, expected_t in zip(result, expected):
            self.assertEqual(result_t, expected_t)

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my_string_op_const_string_ref(self, device):
        """Test my_string_op_const_string_ref which uses const std::string& parameters."""
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libtorch_musa.ops.my_string_op_const_string_ref(
            t, "dim", "test1"
        )
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "test1"])
        self.assertEqual(result_dim, t.dim())

        size_vec, result_size = libtorch_musa.ops.my_string_op_const_string_ref(
            t, "size", "test2"
        )
        self.assertEqual(size_vec, ["size", str(t.size(0)), "test2"])
        self.assertEqual(result_size, t.size(0))

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my_string_op_const_string_view_ref(self, device):
        """Test my_string_op_const_string_view_ref which uses const std::string_view& parameters."""
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libtorch_musa.ops.my_string_op_const_string_view_ref(
            t, "dim", "view1"
        )
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "view1"])
        self.assertEqual(result_dim, t.dim())

        stride_vec, result_stride = (
            libtorch_musa.ops.my_string_op_const_string_view_ref(
                t, "stride", "view2"
            )
        )
        self.assertEqual(stride_vec, ["stride", str(t.stride(0)), "view2"])
        self.assertEqual(result_stride, t.stride(0))

    @skipIfWindows(msg="ValueError: vector too long")
    @skipIfTorchVersionLessThan(2, 10)
    def test_my_string_op_string_ref(self, device):
        """Test my_string_op_string_ref which uses std::string& (non-const) parameters."""
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.empty(3, 4, 5, device=device)

        dim_vec, result_dim = libtorch_musa.ops.my_string_op_string_ref(
            t, "dim", "ref1"
        )
        self.assertEqual(dim_vec, ["dim", str(t.dim()), "ref1"])
        self.assertEqual(result_dim, t.dim())

        size_vec, result_size = libtorch_musa.ops.my_string_op_string_ref(
            t, "size", "ref2"
        )
        self.assertEqual(size_vec, ["size", str(t.size(0)), "ref2"])
        self.assertEqual(result_size, t.size(0))

    @skipIfTorchVersionLessThan(2, 10)
    @onlyCPU
    def test_my_set_requires_grad(self, device):
        """Test set_requires_grad method on Tensor."""
        import libtorch_agn_2_10 as libtorch_musa

        # Use torch.no_grad() to prevent autograd from wrapping the output
        # tensor with a grad_fn. When a tensor with requires_grad=True goes
        # through a custom op, PyTorch wraps the output with a grad_fn
        # (e.g., WarnNotImplemented), making requires_grad computed based on
        # inputs rather than directly settable.
        t = torch.randn(3, 4, device=device)
        self.assertFalse(t.requires_grad)

        with torch.no_grad():
            libtorch_musa.ops.my_set_requires_grad(t, True)
        self.assertTrue(t.requires_grad)

        with torch.no_grad():
            libtorch_musa.ops.my_set_requires_grad(t, False)
        self.assertFalse(t.requires_grad)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_my_get_current_musa_stream(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        device_index = torch.device(device).index
        res = libtorch_musa.ops.my_get_current_musa_stream(device_index)
        expected = torch.musa.current_stream(device_index).musa_stream
        self.assertEqual(res, expected)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_my_set_current_musa_stream(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        device_index = torch.device(device).index
        prev_stream = torch.musa.current_stream(device_index).musa_stream
        new_stream = torch.musa.Stream(device_index).musa_stream

        try:
            libtorch_musa.ops.my_set_current_musa_stream(new_stream, device_index)
            expected = torch.musa.current_stream(device_index).musa_stream
            self.assertEqual(new_stream, expected)
        finally:
            libtorch_musa.ops.my_set_current_musa_stream(prev_stream, device_index)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_my_get_musa_stream_from_pool(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        device_index = torch.device(device).index
        prev_stream = torch.musa.current_stream(device_index).musa_stream

        try:
            for high_priority in [False, True]:
                stream = libtorch_musa.ops.my_get_musa_stream_from_pool(
                    high_priority, device_index
                )
                libtorch_musa.ops.my_set_current_musa_stream(stream, device_index)
                expected = torch.musa.current_stream(device_index).musa_stream
                self.assertEqual(stream, expected)
        finally:
            libtorch_musa.ops.my_set_current_musa_stream(prev_stream, device_index)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_my_musa_stream_synchronize(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        device_index = torch.device(device).index
        stream = torch.musa.current_stream(device_index).musa_stream
        # sanity check for torch_musa_stream_synchronize:
        libtorch_musa.ops.my_musa_stream_synchronize(stream, device_index)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_my_aoti_create_musa_guard(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        previous_device = torch.musa.current_device()
        device_index = torch.device(device).index
        try:
            libtorch_musa.ops.my_aoti_create_musa_guard(device_index)
        finally:
            torch.musa.set_device(previous_device)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_my_aoti_create_musa_stream_guard(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        device_index = torch.device(device).index
        previous_stream = torch.musa.current_stream(device_index).musa_stream
        stream_obj = torch.musa.Stream(device_index)
        stream = stream_obj.musa_stream
        try:
            libtorch_musa.ops.my_aoti_create_musa_stream_guard(stream, device_index)
        finally:
            libtorch_musa.ops.my_set_current_musa_stream(
                previous_stream, device_index
            )

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_my_aoti_musa_caching_allocator_alloc_delete(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        torch.musa.synchronize(device)
        initial_memory = torch.musa.memory_allocated(device)
        self.assertFalse(
            libtorch_musa.ops.my_aoti_musa_caching_allocator_alloc_delete(0)
        )
        self.assertTrue(
            libtorch_musa.ops.my_aoti_musa_caching_allocator_alloc_delete(16)
        )
        torch.musa.synchronize(device)
        self.assertEqual(torch.musa.memory_allocated(device), initial_memory)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_my_aoti_model_base_raii_gpu_malloc(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        torch.musa.synchronize(device)
        initial_memory = torch.musa.memory_allocated(device)
        self.assertFalse(libtorch_musa.ops.my_aoti_model_base_raii_gpu_malloc(0))
        self.assertTrue(libtorch_musa.ops.my_aoti_model_base_raii_gpu_malloc(16))
        torch.musa.synchronize(device)
        self.assertEqual(torch.musa.memory_allocated(device), initial_memory)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_my_aoti_musa_runtime_raii_guard(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        previous_device = torch.musa.current_device()
        device_index = torch.device(device).index
        previous_stream = torch.musa.current_stream(device_index).musa_stream
        stream_obj = torch.musa.Stream(device_index)
        stream = stream_obj.musa_stream
        try:
            libtorch_musa.ops.my_aoti_musa_runtime_raii_guard(stream, device_index)
        finally:
            libtorch_musa.ops.my_set_current_musa_stream(
                previous_stream, device_index
            )
            torch.musa.set_device(previous_device)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_from_blob(self, device):
        import libtorch_agn_2_10 as libtorch_musa

        # Create reference implementation using unstable torch::from_blob via load_inline
        source = """
        #include <torch/extension.h>

        at::Tensor reference_from_blob(at::Tensor t) {
            void* data_ptr = t.storage().data_ptr().get();
            auto options = torch::TensorOptions()
                .dtype(t.dtype())
                .device(t.device());

            return torch::from_blob(
                data_ptr,
                t.sizes(),
                t.strides(),
                options);
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="test_from_blob_reference",
            cpp_sources=[source],
            functions=["reference_from_blob"],
        )

        # Test basic from_blob with contiguous tensor
        original = torch.rand(2, 3, device=device, dtype=torch.float32)
        stable_result = libtorch_musa.ops.my_from_blob(
            original.data_ptr(),
            original.size(),
            original.stride(),
            device,
            torch.float32,
        )
        reference_result = module.reference_from_blob(original)
        self.assertEqual(stable_result, reference_result)
        self.assertEqual(stable_result.data_ptr(), original.data_ptr())

        # Test with non-contiguous strides
        transposed = torch.rand(4, 6, device=device, dtype=torch.float32).t()

        stable_transposed = libtorch_musa.ops.my_from_blob(
            transposed.data_ptr(),
            transposed.size(),
            transposed.stride(),
            device,
            transposed.dtype,
        )

        reference_transposed = module.reference_from_blob(transposed)
        self.assertEqual(stable_transposed, reference_transposed)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_std_musa_check_success(self, device):
        """Test that STD_MUSA_CHECK works correctly for successful MUSA calls."""
        import libtorch_agn_2_10 as libtorch_musa

        result = libtorch_musa.ops.test_std_musa_check_success()
        expected_device = torch.musa.current_device()
        self.assertEqual(result, expected_device)

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    @parametrize("show_cpp_stacktraces", [False, True])
    def test_std_musa_check_error(self, device, show_cpp_stacktraces):
        """Test that STD_MUSA_CHECK throws std::runtime_error with MUSA error message.

        When TORCH_SHOW_CPP_STACKTRACES=1, the error should include a C++ stack trace.
        Since this env var is cached on first use, we use subprocess to test both cases.
        """
        import os
        import subprocess
        import sys

        test_script = """
import torch
import libtorch_agn_2_10 as libtorch_musa

try:
    libtorch_musa.ops.test_std_musa_check_error()
except RuntimeError as e:
    print(str(e))
"""
        env = os.environ.copy()
        env["TORCH_SHOW_CPP_STACKTRACES"] = "1" if show_cpp_stacktraces else "0"
        # Pass the current sys.path to subprocess so it can find the locally installed extension
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            env=env,
        )

        error_message = result.stdout + result.stderr

        self.assertTrue(
            "MUSA error: invalid device ordinal" in error_message,
            f"Expected 'MUSA error: invalid device ordinal' in error message, got: {error_message}",
        )

        if show_cpp_stacktraces:
            self.assertIn("C++ CapturedTraceback:", error_message)
            self.assertIn("test_std_musa_check_error", error_message)
        else:
            self.assertNotIn("C++ CapturedTraceback:", error_message)

    @skipIfTorchVersionLessThan(2, 10)
    @skipIfTorchDynamo(" Dynamo failed to run FX node with fake tensors")
    def test_my_to_device(self, device):
        """Test to(device) convenience overload."""
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(3, 4, device="cpu")

        # Move to current device
        result = libtorch_musa.ops.my_to_device(t, device)
        expected = t.to(device)
        self.assertEqual(result, expected, exact_device=True)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_to_dtype(self, device):
        """Test to(dtype) via the main to function."""
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(3, 4, device=device, dtype=torch.float32)

        # Convert to float64
        result = libtorch_musa.ops.my_to_dtype(t, torch.float64)
        expected = t.to(torch.float64)
        self.assertEqual(result, expected, exact_device=True)

        # Convert to int32
        t2 = torch.randn(2, 3, device=device, dtype=torch.float32)
        result2 = libtorch_musa.ops.my_to_dtype(t2, torch.int32)
        expected2 = t2.to(torch.int32)
        self.assertEqual(result2, expected2, exact_device=True)

    @skipIfTorchVersionLessThan(2, 10)
    @skipIfTorchDynamo(" Dynamo failed to run FX node with fake tensors")
    def test_my_to_dtype_layout(self, device):
        """Test the full to.dtype_layout op with various parameter combinations."""
        import libtorch_agn_2_10 as libtorch_musa

        # Test dtype conversion
        t = torch.randn(3, 4, device=device, dtype=torch.float32)
        result = libtorch_musa.ops.my_to_dtype_layout(t, dtype=torch.float64)
        expected = t.to(dtype=torch.float64)
        self.assertEqual(result, expected, exact_device=True)

        # Test device conversion (move to CPU if on MUSA, or stay on CPU)
        result_cpu = libtorch_musa.ops.my_to_dtype_layout(t, device="cpu")
        expected_cpu = t.to(device="cpu")
        self.assertEqual(result_cpu, expected_cpu, exact_device=True)

        # Test copy=True (should always create a copy)
        t_copy = torch.randn(2, 3, device=device)
        result_copy = libtorch_musa.ops.my_to_dtype_layout(t_copy, copy=True)
        expected_copy = t_copy.to(copy=True)
        self.assertEqual(result_copy, expected_copy, exact_device=True)
        self.assertNotEqual(result_copy.data_ptr(), t_copy.data_ptr())

        # Test dtype + device together
        t3 = torch.randn(2, 2, device=device, dtype=torch.float32)
        result_both = libtorch_musa.ops.my_to_dtype_layout(
            t3, dtype=torch.float64, device="cpu"
        )
        expected_both = t3.to(dtype=torch.float64, device="cpu")
        self.assertEqual(result_both, expected_both, exact_device=True)

        # Test memory_format (channels_last for 4D tensor)
        t4d = torch.randn(2, 3, 4, 5, device=device)
        result_channels_last = libtorch_musa.ops.my_to_dtype_layout(
            t4d, memory_format=torch.channels_last
        )
        expected_channels_last = t4d.to(memory_format=torch.channels_last)
        self.assertEqual(
            result_channels_last, expected_channels_last, exact_device=True
        )
        self.assertTrue(
            result_channels_last.is_contiguous(memory_format=torch.channels_last)
        )

        # Test with all None (should return equivalent tensor)
        t_none = torch.randn(2, 3, device=device)
        result_none = libtorch_musa.ops.my_to_dtype_layout(t_none)
        expected_none = t_none.to()
        self.assertEqual(result_none, expected_none, exact_device=True)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_contiguous(self, device):
        """Test contiguous with default memory format."""
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(3, 4, device=device).t()
        self.assertFalse(t.is_contiguous())

        result = libtorch_musa.ops.my_contiguous(t)
        self.assertTrue(result.is_contiguous())

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_contiguous_memory_format(self, device):
        """Test contiguous with specified memory format."""
        import libtorch_agn_2_10 as libtorch_musa

        # Create a 4D tensor (N, C, H, W)
        t = torch.randn(2, 3, 4, 5, device=device)

        # Convert to channels_last format
        result = libtorch_musa.ops.my_contiguous_memory_format(
            t, torch.channels_last
        )
        self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    def test_std_musa_kernel_launch_check_success(self, device):
        """Test that STD_MUSA_KERNEL_LAUNCH_CHECK works correctly for successful kernel launches."""
        import libtorch_agn_2_10 as libtorch_musa

        libtorch_musa.ops.test_std_musa_kernel_launch_check_success()

    @skipIfTorchVersionLessThan(2, 10)
    @onlyPRIVATEUSE1
    @parametrize("show_cpp_stacktraces", [False, True])
    def test_std_musa_kernel_launch_check_error(self, device, show_cpp_stacktraces):
        """Test that STD_MUSA_KERNEL_LAUNCH_CHECK throws std::runtime_error for invalid kernel launches.

        When TORCH_SHOW_CPP_STACKTRACES=1, the error should include a C++ stack trace.
        Since this env var is cached on first use, we use subprocess to test both cases.
        """
        import os
        import subprocess
        import sys

        test_script = """
import torch
import libtorch_agn_2_10 as libtorch_musa

try:
    libtorch_musa.ops.test_std_musa_kernel_launch_check_error()
except RuntimeError as e:
    print(str(e))
"""
        env = os.environ.copy()
        env["TORCH_SHOW_CPP_STACKTRACES"] = "1" if show_cpp_stacktraces else "0"
        # Pass the current sys.path to subprocess so it can find the locally installed extension
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            env=env,
        )

        error_message = result.stdout + result.stderr

        self.assertTrue(
            "MUSA error: invalid configuration argument" in error_message,
            f"Expected 'MUSA error: invalid configuration argument' in error message, got: {error_message}",
        )

        if show_cpp_stacktraces:
            self.assertIn("C++ CapturedTraceback:", error_message)
            self.assertIn("test_std_musa_kernel_launch_check_error", error_message)
        else:
            self.assertNotIn("C++ CapturedTraceback:", error_message)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_new_empty(self, device):
        """Test new_empty with all kwargs."""
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(3, 4, device=device, dtype=torch.float32)

        # Test with default args (should inherit from self)
        result = libtorch_musa.ops.my_new_empty(t, [2, 3])
        expected = t.new_empty([2, 3])
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.dtype, expected.dtype)
        self.assertEqual(result.device, expected.device)

        # Test with different dtype
        result_dtype = libtorch_musa.ops.my_new_empty(
            t, [2, 3], dtype=torch.float64
        )
        expected_dtype = t.new_empty([2, 3], dtype=torch.float64)
        self.assertEqual(result_dtype.shape, expected_dtype.shape)
        self.assertEqual(result_dtype.dtype, torch.float64)

        # Test with different device (move to CPU)
        result_device = libtorch_musa.ops.my_new_empty(t, [2, 3], device="cpu")
        expected_device = t.new_empty([2, 3], device="cpu")
        self.assertEqual(result_device.shape, expected_device.shape)
        self.assertEqual(result_device.device.type, "cpu")

        # Test with dtype and device together
        result_both = libtorch_musa.ops.my_new_empty(
            t, [4, 5], dtype=torch.int64, device="cpu"
        )
        expected_both = t.new_empty([4, 5], dtype=torch.int64, device="cpu")
        self.assertEqual(result_both.shape, expected_both.shape)
        self.assertEqual(result_both.dtype, torch.int64)
        self.assertEqual(result_both.device.type, "cpu")

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_new_zeros(self, device):
        """Test new_zeros with all kwargs."""
        import libtorch_agn_2_10 as libtorch_musa

        t = torch.randn(3, 4, device=device, dtype=torch.float32)

        # Test with default args (should inherit from self)
        result = libtorch_musa.ops.my_new_zeros(t, [2, 3])
        expected = t.new_zeros([2, 3])
        self.assertEqual(result, expected, exact_device=True)

        # Test with different dtype
        result_dtype = libtorch_musa.ops.my_new_zeros(
            t, [2, 3], dtype=torch.float64
        )
        expected_dtype = t.new_zeros([2, 3], dtype=torch.float64)
        self.assertEqual(result_dtype, expected_dtype, exact_device=True)

        # Test with different device (move to CPU)
        result_device = libtorch_musa.ops.my_new_zeros(t, [2, 3], device="cpu")
        expected_device = t.new_zeros([2, 3], device="cpu")
        self.assertEqual(result_device, expected_device, exact_device=True)

        # Test with dtype and device together
        result_both = libtorch_musa.ops.my_new_zeros(
            t, [4, 5], dtype=torch.int64, device="cpu"
        )
        expected_both = t.new_zeros([4, 5], dtype=torch.int64, device="cpu")
        self.assertEqual(result_both, expected_both, exact_device=True)

    def test_my_unsqueeze(self, device):
        """Test unsqueeze op."""
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.randn(3, 4, device=device)

        # Test unsqueeze at dim 0
        result = libtorch_musa.ops.my_unsqueeze(t, 0)
        expected = torch.unsqueeze(t, 0)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([1, 3, 4]))

        # Test unsqueeze at dim 1
        result1 = libtorch_musa.ops.my_unsqueeze(t, 1)
        expected1 = torch.unsqueeze(t, 1)
        self.assertEqual(result1, expected1)
        self.assertEqual(result1.shape, torch.Size([3, 1, 4]))

        # Test unsqueeze at dim -1
        result_neg = libtorch_musa.ops.my_unsqueeze(t, -1)
        expected_neg = torch.unsqueeze(t, -1)
        self.assertEqual(result_neg, expected_neg)
        self.assertEqual(result_neg.shape, torch.Size([3, 4, 1]))

    def test_my_squeeze(self, device):
        """Test squeeze.dim op."""
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.randn(3, 1, 4, device=device)

        # Test squeeze at dim 1 (the dimension of size 1)
        result = libtorch_musa.ops.my_squeeze(t, 1)
        expected = torch.squeeze(t, 1)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([3, 4]))

        # Test squeeze at dim 0 (not size 1, should be no-op)
        result0 = libtorch_musa.ops.my_squeeze(t, 0)
        expected0 = torch.squeeze(t, 0)
        self.assertEqual(result0, expected0)
        self.assertEqual(result0.shape, torch.Size([3, 1, 4]))

        # Test squeeze at dim -2 (same as dim 1)
        result_neg = libtorch_musa.ops.my_squeeze(t, -2)
        expected_neg = torch.squeeze(t, -2)
        self.assertEqual(result_neg, expected_neg)
        self.assertEqual(result_neg.shape, torch.Size([3, 4]))

    def test_my_select(self, device):
        """Test select.int op."""
        import libtorch_agn_2_9 as libtorch_musa

        t = torch.randn(3, 4, 5, device=device)

        # Test select at dim 0, index 1
        result = libtorch_musa.ops.my_select(t, 0, 1)
        expected = torch.select(t, 0, 1)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([4, 5]))

        # Test select at dim 1, index 2
        result1 = libtorch_musa.ops.my_select(t, 1, 2)
        expected1 = torch.select(t, 1, 2)
        self.assertEqual(result1, expected1)
        self.assertEqual(result1.shape, torch.Size([3, 5]))

        # Test select at dim -1, index 0
        result_neg = libtorch_musa.ops.my_select(t, -1, 0)
        expected_neg = torch.select(t, -1, 0)
        self.assertEqual(result_neg, expected_neg)
        self.assertEqual(result_neg.shape, torch.Size([3, 4]))

    def test_my_matmul(self, device):
        """Test matmul op."""
        import libtorch_agn_2_9 as libtorch_musa

        # Test 2D x 2D matrix multiplication
        a = torch.randn(3, 4, device=device)
        b = torch.randn(4, 5, device=device)
        result = libtorch_musa.ops.my_matmul(a, b)
        expected = torch.matmul(a, b)
        self.assertEqual(result, expected)
        self.assertEqual(result.shape, torch.Size([3, 5]))

        # Test 1D x 2D (vector-matrix)
        v = torch.randn(4, device=device)
        m = torch.randn(4, 5, device=device)
        result_vm = libtorch_musa.ops.my_matmul(v, m)
        expected_vm = torch.matmul(v, m)
        self.assertEqual(result_vm, expected_vm)

        # Test 2D x 1D (matrix-vector)
        m2 = torch.randn(3, 4, device=device)
        v2 = torch.randn(4, device=device)
        result_mv = libtorch_musa.ops.my_matmul(m2, v2)
        expected_mv = torch.matmul(m2, v2)
        self.assertEqual(result_mv, expected_mv)

        # Test batched matmul
        batch_a = torch.randn(2, 3, 4, device=device)
        batch_b = torch.randn(2, 4, 5, device=device)
        result_batch = libtorch_musa.ops.my_matmul(batch_a, batch_b)
        expected_batch = torch.matmul(batch_a, batch_b)
        self.assertEqual(result_batch, expected_batch)

    @skipIfTorchVersionLessThan(2, 10)
    def test_my_subtract(self, device):
        """Test subtract.Tensor op."""
        import libtorch_agn_2_10 as libtorch_musa

        a = torch.randn(3, 4, device=device)
        b = torch.randn(3, 4, device=device)

        # Test basic subtraction (alpha=1.0)
        result = libtorch_musa.ops.my_subtract(a, b)
        expected = torch.subtract(a, b)
        self.assertEqual(result, expected)

        # Test subtraction with alpha=2.0
        result_alpha = libtorch_musa.ops.my_subtract(a, b, alpha=2.0)
        expected_alpha = torch.subtract(a, b, alpha=2.0)
        self.assertEqual(result_alpha, expected_alpha)

        # Test subtraction with alpha=0.5
        result_half = libtorch_musa.ops.my_subtract(a, b, alpha=0.5)
        expected_half = torch.subtract(a, b, alpha=0.5)
        self.assertEqual(result_half, expected_half)

        # Test subtraction with broadcasting
        c = torch.randn(4, device=device)
        result_broadcast = libtorch_musa.ops.my_subtract(a, c)
        expected_broadcast = torch.subtract(a, c)
        self.assertEqual(result_broadcast, expected_broadcast)

    @skipIfTorchVersionLessThan(2, 11)
    def test_my_from_blob_with_deleter(self, device):
        """Test for from_blob with custom deleter (2.11 feature)."""
        import libtorch_agn_2_11 as libtorch_musa

        is_musa = torch.device(device).type == "musa"
        if is_musa:
            init_mem = torch.musa.memory_allocated(device)

        def inner():
            libtorch_musa.ops.reset_deleter_call_count()
            self.assertEqual(libtorch_musa.ops.get_deleter_call_count(), 0)

            # We need an original tensor to create the tensor with from_blob.
            original = torch.rand(2, 3, device=device, dtype=torch.float32)
            blob_tensor = libtorch_musa.ops.my_from_blob_with_deleter(
                original.data_ptr(),
                original.size(),
                original.stride(),
                device,
                torch.float32,
            )

            self.assertEqual(blob_tensor, original)
            self.assertEqual(blob_tensor.data_ptr(), original.data_ptr())

            self.assertEqual(libtorch_musa.ops.get_deleter_call_count(), 0)

            del blob_tensor
            gc.collect()

            # Ensure the deleter was called. The original tensor still exists
            # and can be used.
            self.assertEqual(libtorch_musa.ops.get_deleter_call_count(), 1)
            original += 1
            # original goes out of scope here and its MUSA memory should be
            # freed.

        inner()

        if is_musa:
            # original tensor is out of scope, all the memory should be freed
            torch.musa.synchronize(device)
            curr_mem = torch.musa.memory_allocated(device)
            self.assertEqual(curr_mem, init_mem)

    @onlyPRIVATEUSE1
    @skipIfTorchVersionLessThan(2, 11)
    def test_my_from_blob_with_musa_deleter_no_leak(self, device):
        """Test that from_blob deleter properly frees musaMalloc'd memory."""
        import libtorch_agn_2_11 as libtorch_musa

        torch.musa.synchronize(device)
        init_mem = torch.musa.memory_allocated(device)
        numel = 1024 * 1024  # 4 MB per tensor

        for _ in range(10):
            tensor = libtorch_musa.ops.my_from_blob_with_musa_deleter(numel, device)
            # Verify tensor was created correctly
            self.assertEqual(tensor.numel(), numel)
            self.assertEqual(tensor.device, torch.device(device))
            del tensor
            gc.collect()
            torch.musa.synchronize(device)

            curr_mem = torch.musa.memory_allocated(device)
            self.assertEqual(curr_mem, init_mem)

    @skipIfTorchVersionLessThan(2, 11)
    def test_my_from_blob_with_lambda_deleter(self, device):
        """Test for from_blob with capturing-lambda deleter (2.11 feature)."""
        import libtorch_agn_2_11 as libtorch_musa

        from_blob_fn = libtorch_musa.ops.my_from_blob_with_lambda_deleter
        get_count = libtorch_musa.ops.get_lambda_deleter_call_count
        reset_count = libtorch_musa.ops.reset_lambda_deleter_call_count

        is_musa = torch.device(device).type == "musa"
        if is_musa:
            init_mem = torch.musa.memory_allocated(device)

        def inner():
            reset_count()
            self.assertEqual(get_count(), 0)

            # We need an original tensor to create the tensor with from_blob.
            original = torch.rand(2, 3, device=device, dtype=torch.float32)
            blob_tensor = from_blob_fn(
                original.data_ptr(),
                original.size(),
                original.stride(),
                device,
                torch.float32,
            )

            self.assertEqual(blob_tensor, original)
            self.assertEqual(blob_tensor.data_ptr(), original.data_ptr())

            self.assertEqual(get_count(), 0)

            del blob_tensor
            gc.collect()

            # Ensure the deleter was called. The original tensor still exists
            # and can be used.
            self.assertEqual(get_count(), 1)
            original += 1
            # original goes out of scope here and its MUSA memory should be
            # freed.

        inner()

        if is_musa:
            # original tensor is out of scope, all the memory should be freed
            torch.musa.synchronize(device)
            curr_mem = torch.musa.memory_allocated(device)
            self.assertEqual(curr_mem, init_mem)

    @onlyPRIVATEUSE1
    @skipIfTorchVersionLessThan(2, 11)
    def test_my_from_blob_with_musa_lambda_deleter_no_leak(self, device):
        """Test that from_blob lambda deleter properly frees musaMalloc'd memory."""
        import libtorch_agn_2_11 as libtorch_musa

        from_blob_fn = libtorch_musa.ops.my_from_blob_with_musa_lambda_deleter

        torch.musa.synchronize(device)
        init_mem = torch.musa.memory_allocated(device)
        numel = 1024 * 1024  # 4 MB per tensor

        for _ in range(10):
            tensor = from_blob_fn(numel, device)
            # Verify tensor was created correctly
            self.assertEqual(tensor.numel(), numel)
            self.assertEqual(tensor.device, torch.device(device))
            del tensor
            gc.collect()
            torch.musa.synchronize(device)

            curr_mem = torch.musa.memory_allocated(device)
            self.assertEqual(curr_mem, init_mem)

    @onlyCPU
    def test_my_layout(self, device):
        """Test layout() method for various tensor layouts."""
        import libtorch_agn_2_9 as libtorch_musa

        # Test strided layout
        t_strided = torch.randn(3, 4, device=device)
        self.assertTrue(libtorch_musa.ops.my_layout(t_strided, torch.strided))
        self.assertFalse(libtorch_musa.ops.my_layout(t_strided, torch.sparse_coo))

        # Test sparse COO layout
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
        values = torch.tensor([1.0, 2.0, 3.0])
        t_sparse_coo = torch.sparse_coo_tensor(indices, values, (3, 3))
        self.assertTrue(libtorch_musa.ops.my_layout(t_sparse_coo, torch.sparse_coo))
        self.assertFalse(libtorch_musa.ops.my_layout(t_sparse_coo, torch.strided))

        # Test sparse CSR layout
        crow_indices = torch.tensor([0, 1, 2, 3])
        col_indices = torch.tensor([0, 1, 2])
        csr_values = torch.tensor([1.0, 2.0, 3.0])
        t_sparse_csr = torch.sparse_csr_tensor(crow_indices, col_indices, csr_values)
        self.assertTrue(libtorch_musa.ops.my_layout(t_sparse_csr, torch.sparse_csr))
        self.assertFalse(libtorch_musa.ops.my_layout(t_sparse_csr, torch.strided))


instantiate_device_type_tests(
    TestLibtorchMusa, globals(), only_for=("cpu", "privateuse1")
)

if __name__ == "__main__":
    run_tests()
