"""Test musa-converter"""

import os
import subprocess
import shutil


class TestMUSAConverter:
    """class of TestMUSAConverter"""

    # ignore PytestCollectionWarning
    __test__ = False

    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.work_dir = os.path.join(cur_dir, "musa_converter_test")
        os.makedirs(self.work_dir, exist_ok=True)

    def generate_source_code(self):
        """generate python code"""
        python_src = """
import torch
import torch.distributed as dist
from torch.distributed import is_nccl_available
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension \\
    import CUDA_HOME

os.environ["CUDA_VISIBLE_DEVICES"]
is_nccl_avai = is_nccl_available()
backend = random.choice(["nccl", "NCCL"])
backend = Backend.NCCL
dist.init_process_group(backend=backend)

from torch.cuda import amp
import torch.cuda

tensor = torch.randn((32, 32), device="cuda").cuda()
with amp.autocast(enabled=True):
    out = torch.exp(tensor)
assert out.is_cuda
assert torch.cuda._initialized

is_cuda = torch.cuda.is_available()
f_string = f"Use CUDA: {is_cuda}"
"""

        python_dst = """
import torch_musa
import torch
import torch.distributed as dist
from torch.distributed import is_mccl_available
from torch_musa.utils.musa_extension import MUSA_HOME
from torch_musa.utils.musa_extension \\
    import MUSA_HOME

os.environ["MUSA_VISIBLE_DEVICES"]
is_mccl_avai = is_mccl_available()
backend = random.choice(["mccl", "MCCL"])
backend = Backend.MCCL
dist.init_process_group(backend=backend)

from torch_musa import amp
import torch_musa

tensor = torch.randn((32, 32), device="musa").musa()
with amp.autocast(enabled=True):
    out = torch.exp(tensor)
assert out.is_musa
assert torch.musa._initialized

is_musa = torch.musa.is_available()
f_string = f"Use MUSA: {is_musa}"
"""
        with open(f"{self.work_dir}/python_src.py", "w", encoding="utf-8") as f:
            f.write(python_src)

        return python_src, python_dst

    def run(self):
        """run unit test"""
        _, python_dst = self.generate_source_code()
        ori_dir = os.getcwd()
        os.chdir(self.work_dir)
        try:
            subprocess.check_call(
                [
                    "musa-converter",
                    "-r",
                    f"{self.work_dir}",
                    "-l",
                    f"{self.work_dir}/python_src.py",
                ]
            )
        except subprocess.CalledProcessError:
            assert False, "musa-converter command failed"

        with open(f"{self.work_dir}/python_src.py", "r", encoding="utf-8") as f:
            python_src = f.read()

        assert (
            python_src == python_dst
        ), "There maybe something wrong with the musa_converter tool"

        os.chdir(ori_dir)
        shutil.rmtree(self.work_dir)


def test_musa_converter():
    TestMUSAConverter().run()
