"""Security tests for torch_musa.utils.compare_tool."""

import os
from pathlib import Path
import pickle
import shlex

import pytest
import torch

from torch_musa import testing
from torch_musa.utils import compare_tool


def _normal_add(left, offset):
    return left + offset


def _slice_add(left, index, right):
    return _normal_add(left[index], right)


@testing.skip_if_musa_unavailable
def test_compare_tool_basic(tmp_path):
    """Saved tensor arguments can be loaded and compared again."""
    args = (torch.arange(4, dtype=torch.float32, device="musa"),)
    kwargs = {"offset": 1}
    output = _normal_add(args[0], kwargs["offset"])
    compare_tool.save_data_for_op(
        output, args, kwargs, str(tmp_path), "basic", file_suffix=""
    )

    correct, loaded_args, loaded_kwargs, loaded_output = (
        compare_tool.compare_for_single_op(
            str(tmp_path / "basic_inputs.pt"),
            _normal_add,
            atol=0,
            rtol=0,
        )
    )

    assert correct
    assert torch.equal(loaded_args[0], args[0])
    assert loaded_args[0].device.type == "musa"
    assert loaded_kwargs == kwargs
    assert torch.equal(loaded_output, output)


@testing.skip_if_musa_unavailable
def test_compare_tool_with_slice(tmp_path):
    """Indexing arguments containing ``slice`` remain usable after loading."""
    args = (
        torch.arange(6, dtype=torch.float32, device="musa"),
        slice(1, 5, 2),
        1,
    )
    output = _slice_add(*args)
    compare_tool.save_data_for_op(
        output, args, {}, str(tmp_path), "slice", file_suffix=""
    )

    correct, loaded_args, _, loaded_output = compare_tool.compare_for_single_op(
        str(tmp_path / "slice_inputs.pt"), _slice_add, atol=0, rtol=0
    )

    assert correct
    assert loaded_args[0].device.type == "musa"
    assert loaded_args[1] == slice(1, 5, 2)
    assert loaded_args[2] == 1
    assert torch.equal(loaded_output, output)


class UnsafeModule:
    """Payload that would create a marker if unrestricted Pickle ran it."""

    def __init__(self, path: Path):
        self.path = path

    def __reduce__(self):
        cmd = f"sh -c 'echo UnsafeModule > {shlex.quote(self.path)}'"
        return (os.system, (cmd,))


@pytest.mark.parametrize(
    "loader",
    [compare_tool.compare_for_single_op, compare_tool.nan_inf_track_for_single_op],
)
@pytest.mark.parametrize("saver", ["pickle", "torch"])
def test_compare_tool_rejects_malicious_payload(tmp_path, loader, saver):
    """Restricted loading rejects a malicious global without executing it."""
    marker_path = tmp_path / "marker.txt"
    if marker_path.exists():
        marker_path.unlink()
    data = {
        "args": (),
        "kwargs": {},
        "other": UnsafeModule(str(marker_path)),
    }

    if saver == "pickle":
        checkpoint = tmp_path / "test.pkl"
        with open(checkpoint, "wb") as f:
            pickle.dump(data, f, protocol=2)
    else:
        checkpoint = tmp_path / "test.pt"
        torch.save(data, checkpoint)
    assert not marker_path.exists()

    with pytest.raises(pickle.UnpicklingError, match="unsupported GLOBAL"):
        if loader is compare_tool.compare_for_single_op:
            loader(str(checkpoint), torch.ops.aten.addmm, atol=0.01, rtol=0.01)
        else:
            loader(str(checkpoint), torch.ops.aten.addmm)
    assert not marker_path.exists()
