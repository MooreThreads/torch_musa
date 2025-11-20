"""using a simple tensor add example to verify the usability of MUSAExtension"""

# pylint: disable=missing-function-docstring, unused-import
import os
import importlib
import shutil
import torch
import torch_musa


class TestMUSAExtension:
    """class of TestMUSAExtension"""

    # ignore PytestCollectionWarning
    __test__ = False

    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.work_dir = os.path.join(cur_dir, "musa_extension_test")
        os.makedirs(self.work_dir, exist_ok=True)

    def generate_tensor_add_source_code(self):
        cpp_src = """
#include <torch/torch.h>
#include <torch/extension.h>
at::Tensor musa_add(
    at::Tensor & a,
    at::Tensor & b);
#define CHECK_MUSA(x) AT_ASSERTM(x.is_privateuseone(), #x " must be a MUSA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_MUSA(x); CHECK_CONTIGUOUS(x)
at::Tensor add(
    at::Tensor & a,
    at::Tensor & b ){
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    return musa_add(a, b);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("musa_add", &add, "add op example");
}
        """

        mu_src = """
#include <ATen/ATen.h>
#include <musa.h>
#include <musa_runtime.h>
#include <torch/extension.h>
__device__ __host__ __forceinline__ constexpr int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}
template <typename T>
__global__ void musa_add_kernel(const T* a, const T* b, T* out, int64_t numel) {
    int stride = blockDim.x * gridDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < numel; i += stride) {
        out[i] = a[i] + b[i];
    }
}
at::Tensor musa_add(at::Tensor& a, at::Tensor& b) {
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1);
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1);
    at::Tensor out = torch::empty(a.sizes(), a.options().device(torch::kPrivateUse1));
    int numel = a.numel();
    const int block_size = 1024;
    int block_num = std::min(ceil_div(numel, block_size), 16);
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "musa_add_kernel", ([&] {
        musa_add_kernel<scalar_t> <<<block_num, block_size, 0>>>(
            a.const_data_ptr<scalar_t>(),
            b.const_data_ptr<scalar_t>(),
            out.mutable_data_ptr<scalar_t>(),
            numel
        );
    }));
    return out;
}
        """

        setup_src = """
from setuptools import setup
import setuptools.command.install
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension
# hacky way for putting *cpython*.so into the root directory of site-packages
class Install(setuptools.command.install.install):
    def run(self):
        super().run()
setup(
    name="musa_extension_example",
    ext_modules=[
        MUSAExtension(
            "musa_extension_example",
            [
                "add.cpp",
                "add_kernel.mu",
            ],
            libraries=["musart", "musa"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension, "install": Install},
)
        """
        with open(f"{self.work_dir}/setup.py", "w", encoding="utf-8") as f:
            f.write(setup_src)

        with open(f"{self.work_dir}/add.cpp", "w", encoding="utf-8") as f:
            f.write(cpp_src)

        with open(f"{self.work_dir}/add_kernel.mu", "w", encoding="utf-8") as f:
            f.write(mu_src)

    def run(self):
        self.generate_tensor_add_source_code()
        ori_dir = os.getcwd()
        os.chdir(self.work_dir)
        os.system("python setup.py install")
        os.chdir(ori_dir)
        musa_extension_example = importlib.import_module("musa_extension_example")

        x = torch.randn((4, 32, 32), device="musa")
        y = torch.randn((4, 32, 32), device="musa")
        res = musa_extension_example.musa_add(x, y)
        golden = x + y
        assert torch.allclose(res, golden)

        shutil.rmtree(self.work_dir)

    def load(self):
        self.generate_tensor_add_source_code()
        musa_ext = importlib.import_module("torch_musa.utils.musa_extension")
        musa_extension_example = musa_ext.load(
            name="musa_extension_example",
            sources=[f"{self.work_dir}/add.cpp", f"{self.work_dir}/add_kernel.mu"],
            extra_cflags=[],
            extra_musa_cflags=[],
            verbose=True,
        )

        x = torch.randn((4, 32, 32), device="musa")
        y = torch.randn((4, 32, 32), device="musa")
        res = musa_extension_example.musa_add(x, y)
        golden = x + y
        assert torch.allclose(res, golden)

        shutil.rmtree(self.work_dir)


def test_musa_extension():
    TestMUSAExtension().run()
    TestMUSAExtension().load()
