# pylint: disable=all
import os
import pytest
import torch
import torch_musa

if torch.musa.is_available():
    torch.musa.memory._set_allocator_settings("expandable_segments:True")
    os.environ["PYTORCH_MUSA_ALLOC_CONF"] = "expandable_segments:True"


# DO NOT delete this non op test, otherwise pytest will return non zero ret code
# when torch.version.musa < 4000
# @pytest.mark.skipif(True, reason="")
def test_nonop():
    pass


# 0408 daily core dumped
# @pytest.mark.skipif(True, reason="")
def test_expandable_segments():
    if float(torch.version.musa) >= 4000:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, "../core/test_memory.py")
        exec(compile(open(filepath, "r").read(), filepath, mode="exec"))
