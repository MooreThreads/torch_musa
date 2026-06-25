"""Test _grouped_mm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, C0103
import pytest
import torch

from torch_musa import testing


def _manual_grouped_mm(mat_a, mat_b, offs, out_dtype=None):
    out_dtype = out_dtype or mat_a.dtype
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2

    if a_is_2d and not b_is_2d:
        outputs = []
        group_start_idx = 0
        for group_end_idx, b_slice in zip(offs.tolist(), mat_b):
            a_slice = mat_a[group_start_idx:group_end_idx, :]
            outputs.append(torch.mm(a_slice, b_slice))
            group_start_idx = group_end_idx
        return torch.cat(outputs, dim=0).to(out_dtype)

    if not a_is_2d and b_is_2d:
        outputs = []
        group_start_idx = 0
        for group_idx, group_end_idx in enumerate(offs.tolist()):
            b_slice = mat_b[:, group_start_idx:group_end_idx]
            outputs.append(torch.mm(mat_a[group_idx], b_slice))
            group_start_idx = group_end_idx
        return torch.cat(outputs, dim=1).to(out_dtype)

    if a_is_2d and b_is_2d:
        outputs = []
        group_start_idx = 0
        for group_end_idx in offs.tolist():
            a_slice = mat_a[:, group_start_idx:group_end_idx]
            b_slice = mat_b[group_start_idx:group_end_idx, :]
            outputs.append(torch.mm(a_slice, b_slice))
            group_start_idx = group_end_idx
        return torch.stack(outputs).to(out_dtype)

    return torch.bmm(mat_a, mat_b).to(out_dtype)


def _slice_group_scale(scale, group_idx, start, end):
    if scale.numel() == 1:
        return scale
    if scale.dim() == 1:
        return scale[start:end]
    return scale[group_idx]


def _manual_scaled_grouped_mm(mat_a, mat_b, scale_a, scale_b, offs):
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    groups = mat_b.size(0) if a_is_2d and not b_is_2d else mat_a.size(0)
    if a_is_2d and b_is_2d:
        groups = offs.numel()

    outputs = []
    group_start = 0
    for group_idx in range(groups):
        if a_is_2d and not b_is_2d:
            group_end = offs[group_idx].item()
            a_slice = mat_a[group_start:group_end, :]
            b_slice = mat_b[group_idx]
            sa = _slice_group_scale(scale_a, -1, group_start, group_end).reshape(-1, 1)
            sb = _slice_group_scale(scale_b, group_idx, -1, -1).reshape(1, -1)
            group_start = group_end
        elif not a_is_2d and b_is_2d:
            group_end = offs[group_idx].item()
            a_slice = mat_a[group_idx]
            b_slice = mat_b[:, group_start:group_end]
            sa = _slice_group_scale(scale_a, group_idx, -1, -1).reshape(-1, 1)
            sb = _slice_group_scale(scale_b, -1, group_start, group_end).reshape(1, -1)
            group_start = group_end
        elif a_is_2d and b_is_2d:
            group_end = offs[group_idx].item()
            a_slice = mat_a[:, group_start:group_end]
            b_slice = mat_b[group_start:group_end, :]
            s_start = group_idx * mat_a.size(0)
            s_end = s_start + mat_a.size(0)
            sa = _slice_group_scale(scale_a, -1, s_start, s_end).reshape(-1, 1)
            s_start = group_idx * mat_b.size(1)
            s_end = s_start + mat_b.size(1)
            sb = _slice_group_scale(scale_b, -1, s_start, s_end).reshape(1, -1)
            group_start = group_end
        else:
            a_slice = mat_a[group_idx]
            b_slice = mat_b[group_idx]
            sa = _slice_group_scale(scale_a, group_idx, -1, -1).reshape(-1, 1)
            sb = _slice_group_scale(scale_b, group_idx, -1, -1).reshape(1, -1)
        outputs.append(torch.mm(a_slice.float() * sa, b_slice.float() * sb))

    if a_is_2d and not b_is_2d:
        return torch.cat(outputs, dim=0).to(torch.bfloat16)
    if not a_is_2d and b_is_2d:
        return torch.cat(outputs, dim=1).to(torch.bfloat16)
    return torch.stack(outputs).to(torch.bfloat16)


def _make_2d(shape, dtype, transposed=False):
    if transposed:
        return torch.randn(shape[1], shape[0], dtype=dtype).t()
    return torch.randn(*shape, dtype=dtype)


def _make_3d(shape, dtype, transposed=False):
    if transposed:
        return torch.randn(shape[0], shape[2], shape[1], dtype=dtype).transpose(-2, -1)
    return torch.randn(*shape, dtype=dtype)


GROUPED_MM_CASES = [
    {
        "name": "2d_2d",
        "a_shape": (32, 48),
        "b_shape": (48, 64),
        "offs": [16, 32, 48],
        "a_dim": 2,
        "b_dim": 2,
        "a_transposed": False,
        "b_transposed": True,
    },
    {
        "name": "2d_3d",
        "a_shape": (64, 32),
        "b_shape": (3, 32, 48),
        "offs": [16, 48, 64],
        "a_dim": 2,
        "b_dim": 3,
        "a_transposed": False,
        "b_transposed": True,
    },
    {
        "name": "3d_2d",
        "a_shape": (3, 32, 48),
        "b_shape": (48, 80),
        "offs": [16, 48, 80],
        "a_dim": 3,
        "b_dim": 2,
        "a_transposed": False,
        "b_transposed": True,
    },
    {
        "name": "3d_3d",
        "a_shape": (3, 32, 48),
        "b_shape": (3, 48, 64),
        "offs": None,
        "a_dim": 3,
        "b_dim": 3,
        "a_transposed": False,
        "b_transposed": True,
    },
]


def _get_inputs(case, dtype):
    if case["a_dim"] == 2:
        mat_a = _make_2d(case["a_shape"], dtype, case["a_transposed"])
    else:
        mat_a = _make_3d(case["a_shape"], dtype, case["a_transposed"])

    if case["b_dim"] == 2:
        mat_b = _make_2d(case["b_shape"], dtype, case["b_transposed"])
    else:
        mat_b = _make_3d(case["b_shape"], dtype, case["b_transposed"])

    offs = (
        torch.tensor(case["offs"], dtype=torch.int32)
        if case["offs"] is not None
        else None
    )
    return mat_a, mat_b, offs


def _get_comparator(dtype):
    if dtype == torch.float16:
        return testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3)
    if dtype == torch.bfloat16:
        return testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-2)
    return testing.DefaultComparator(abs_diff=5e-5, rel_diff=1e-5)


def _make_grouped_scales(case):
    groups = len(case["offs"]) if case["offs"] is not None else case["a_shape"][0]
    if case["a_dim"] == 2 and case["b_dim"] == 2:
        scale_a = torch.rand(case["a_shape"][0] * groups, dtype=torch.float32) + 0.5
        scale_b = torch.rand(case["b_shape"][1] * groups, dtype=torch.float32) + 0.5
    elif case["a_dim"] == 2 and case["b_dim"] == 3:
        scale_a = torch.rand(case["a_shape"][0], dtype=torch.float32) + 0.5
        scale_b = (
            torch.rand(case["b_shape"][0], case["b_shape"][2], dtype=torch.float32)
            + 0.5
        )
    elif case["a_dim"] == 3 and case["b_dim"] == 2:
        scale_a = (
            torch.rand(case["a_shape"][0], case["a_shape"][1], dtype=torch.float32)
            + 0.5
        )
        scale_b = torch.rand(case["b_shape"][1], dtype=torch.float32) + 0.5
    else:
        scale_a = (
            torch.rand(case["a_shape"][0], case["a_shape"][1], dtype=torch.float32)
            + 0.5
        )
        scale_b = (
            torch.rand(case["b_shape"][0], case["b_shape"][2], dtype=torch.float32)
            + 0.5
        )
    return scale_a, scale_b


def _quantize_grouped_fp8(mat_a, mat_b, scale_a, scale_b, offs):
    fp_dtype = torch.float8_e4m3fn
    a_fp8 = torch.empty_like(mat_a, dtype=fp_dtype)
    b_fp8 = torch.empty_like(mat_b, dtype=fp_dtype)

    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    groups = mat_b.size(0) if a_is_2d and not b_is_2d else mat_a.size(0)
    if a_is_2d and b_is_2d:
        groups = offs.numel()

    group_start_idx = 0
    for group_idx in range(groups):
        if a_is_2d and not b_is_2d:
            group_end_idx = offs[group_idx].item()
            sa = _slice_group_scale(scale_a, -1, group_start_idx, group_end_idx)
            sb = _slice_group_scale(scale_b, group_idx, -1, -1)
            a_fp8[group_start_idx:group_end_idx, :] = (
                mat_a[group_start_idx:group_end_idx, :] / sa.reshape(-1, 1)
            ).to(fp_dtype)
            b_fp8[group_idx] = (mat_b[group_idx] / sb.reshape(1, -1)).to(fp_dtype)
            group_start_idx = group_end_idx
        elif not a_is_2d and b_is_2d:
            group_end_idx = offs[group_idx].item()
            sa = _slice_group_scale(scale_a, group_idx, -1, -1)
            sb = _slice_group_scale(scale_b, -1, group_start_idx, group_end_idx)
            a_fp8[group_idx] = (mat_a[group_idx] / sa.reshape(-1, 1)).to(fp_dtype)
            b_fp8[:, group_start_idx:group_end_idx] = (
                mat_b[:, group_start_idx:group_end_idx] / sb.reshape(1, -1)
            ).to(fp_dtype)
            group_start_idx = group_end_idx
        elif a_is_2d and b_is_2d:
            group_end_idx = offs[group_idx].item()
            sa = _slice_group_scale(
                scale_a, -1, group_idx * mat_a.size(0), (group_idx + 1) * mat_a.size(0)
            )
            sb = _slice_group_scale(
                scale_b, -1, group_idx * mat_b.size(1), (group_idx + 1) * mat_b.size(1)
            )
            a_fp8[:, group_start_idx:group_end_idx] = (
                mat_a[:, group_start_idx:group_end_idx] / sa.reshape(-1, 1)
            ).to(fp_dtype)
            b_fp8[group_start_idx:group_end_idx, :] = (
                mat_b[group_start_idx:group_end_idx, :] / sb.reshape(1, -1)
            ).to(fp_dtype)
            group_start_idx = group_end_idx
        else:
            sa = _slice_group_scale(scale_a, group_idx, -1, -1)
            sb = _slice_group_scale(scale_b, group_idx, -1, -1)
            a_fp8[group_idx] = (mat_a[group_idx] / sa.reshape(-1, 1)).to(fp_dtype)
            b_fp8[group_idx] = (mat_b[group_idx] / sb.reshape(1, -1)).to(fp_dtype)

    return a_fp8, b_fp8


@pytest.mark.parametrize("case", GROUPED_MM_CASES, ids=lambda x: x["name"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_out_dtype", [False, True])
def test_grouped_mm(case, dtype, use_out_dtype):
    if dtype == torch.bfloat16 and testing.get_musa_arch() < 22:
        pytest.skip("bf16 is not supported on arch less than 22")

    mat_a, mat_b, offs = _get_inputs(case, dtype)
    expected_dtype = dtype
    reference_dtype = torch.float32 if dtype != torch.float32 else dtype
    expected = _manual_grouped_mm(
        mat_a.to(reference_dtype),
        mat_b.to(reference_dtype),
        offs,
        out_dtype=expected_dtype,
    )

    musa_out = torch._grouped_mm(
        mat_a.musa(),
        mat_b.musa(),
        offs.musa() if offs is not None else None,
        None,
        dtype if use_out_dtype else None,
    )

    comparator = _get_comparator(dtype)
    assert musa_out.dtype == expected_dtype
    assert comparator(expected, musa_out.cpu())


@pytest.mark.skipif(
    testing.get_musa_arch() < 31, reason="not supported on arch less than 31"
)
@pytest.mark.parametrize("case", GROUPED_MM_CASES, ids=lambda x: x["name"])
def test_scaled_grouped_mm_fp8_e4m3(case):
    mat_a_fp32, mat_b_fp32, offs = _get_inputs(case, torch.float32)
    scale_a, scale_b = _make_grouped_scales(case)
    print(scale_a.shape)
    mat_a_fp8, mat_b_fp8 = _quantize_grouped_fp8(
        mat_a_fp32, mat_b_fp32, scale_a, scale_b, offs
    )

    expected = _manual_scaled_grouped_mm(mat_a_fp8, mat_b_fp8, scale_a, scale_b, offs)
    musa_out = torch._scaled_grouped_mm(
        mat_a_fp8.musa(),
        mat_b_fp8.musa(),
        scale_a.musa(),
        scale_b.musa(),
        offs.musa() if offs is not None else None,
        None,
        None,
        torch.bfloat16,
        False,
    )

    comparator = _get_comparator(musa_out.dtype)
    assert comparator(expected.float(), musa_out.cpu().float())
