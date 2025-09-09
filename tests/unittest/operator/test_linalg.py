"""Test linear algebra operators."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-import,not-callable,invalid-name
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = [
    torch.randn(16, 100, 16, 16),
    torch.randn(8, 1, 8, 8),
    torch.randn(25, 53, 6, 42, 3),
    torch.randn(25, 5, 65, 42, 3, 6),
    torch.randn(2, 53, 65, 6, 3, 6, 10),
    torch.randn(8, 1, 8, 8).to(memory_format=torch.channels_last),
    torch.randn(8, 4, 1, 1).to(memory_format=torch.channels_last),
    torch.randn(0, 0, 0, 0),
    torch.randn(8, 0, 8, 0).to(memory_format=torch.channels_last),
]
dim = [0, 1, 2, 3]
order = [0, 1, 2, 3, 4]
vector_norm_dtype = [torch.float32]
# if testing.get_musa_arch() >= 22:
#     vector_norm_dtype.append(torch.float64)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dim", dim)
@pytest.mark.parametrize("order", order)
@pytest.mark.parametrize("dtype", vector_norm_dtype)
def test_linalg_vector_norm(input_data, dim, order, dtype):
    m = torch.linalg.vector_norm
    input_data = input_data.to(dtype).requires_grad_()
    output = m(input_data, order, dim)
    musa_input = input_data.to("musa").requires_grad_()
    output_musa = m(musa_input, order, dim)
    res = testing.DefaultComparator(abs_diff=1e-5, rel_diff=1e-4)(
        output, output_musa.cpu()
    )
    assert output.grad_fn.__class__ == output_musa.grad_fn.__class__
    info_str = ""
    if not res:
        atol, rtol, equal_nan = testing.DefaultComparator(
            abs_diff=1e-5, rel_diff=1e-4
        ).get_tolerance()
        mask_t = ~torch.isclose(output_musa.cpu(), output, rtol, atol, equal_nan)
        selected = torch.abs(output[mask_t] - output_musa.cpu()[mask_t])
        info_str = f"Max abs error: {selected.max().item()}"

    assert res, info_str


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", [torch.randn(4, 4), torch.randn(2, 3, 4, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_inv(input_data, dtype):
    m = torch.linalg.inv
    input_data = input_data.to(dtype)
    output = m(input_data)
    musa_input = input_data.to("musa")
    output_musa = m(musa_input)
    assert testing.DefaultComparator(abs_diff=1e-5)(output, output_musa.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", [torch.randn(4, 4), torch.randn(2, 3, 4, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse(input_data, dtype):
    m = torch.inverse
    input_data = input_data.to(dtype)
    output = m(input_data)
    musa_input = input_data.to("musa")
    output_musa = m(musa_input)
    assert testing.DefaultComparator(abs_diff=1e-5, rel_diff=1e-4)(
        output, output_musa.cpu()
    )


inverse_origin_input_list = [
    [
        [0.5000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.5000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ],
    [
        [0.5000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.5000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ],
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", [torch.tensor(inverse_origin_input_list)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse_from_nocontiguous(input_data, dtype):
    m = torch.inverse
    input_data = input_data.to(dtype)
    output = m(input_data[..., :3, :3])
    musa_input = input_data.to("musa")
    output_musa = m(musa_input[..., :3, :3])
    assert testing.DefaultComparator(abs_diff=1e-5, rel_diff=1e-4)(
        output, output_musa.cpu()
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data", [{"A": torch.randn(1, 3, 3), "B": torch.randn(2, 3, 3)}]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_lstsq(input_data, dtype):
    m = torch.linalg.lstsq
    input_data["A"] = input_data["A"].to(dtype)
    input_data["B"] = input_data["B"].to(dtype)
    output = m(input_data["A"].clone(), input_data["B"].clone())
    output_musa = m(
        input_data["A"].to("musa").clone(), input_data["B"].to("musa").clone()
    )
    assert testing.DefaultComparator(abs_diff=1e-5)(
        output.solution, output_musa.solution.cpu()
    )
    assert testing.DefaultComparator(abs_diff=1e-5)(
        output.residuals, output_musa.residuals.cpu()
    )
    assert testing.DefaultComparator(abs_diff=1e-5)(output.rank, output_musa.rank.cpu())
    assert testing.DefaultComparator(abs_diff=1e-5)(
        output.singular_values, output_musa.singular_values.cpu()
    )


input_data = [
    torch.randn(5, 5),
    torch.randn(20, 20),
    torch.randn(1, 3, 3),
    torch.randn(4, 7, 7),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_linalg_cholesky(input_data):
    m = torch.linalg.cholesky
    input_data = input_data @ input_data.mT + 1e-3
    input_musa = input_data.musa()
    output = m(input_data)
    output_musa = m(input_musa)
    assert testing.DefaultComparator(abs_diff=1e-5)(output, output_musa)


input_data = [
    torch.randn(5, 5),
    torch.randn(20, 20),
    torch.randn(18, 18),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
def test_cholesky_inverse(input_data):
    m = torch.cholesky_inverse
    inp = torch.mm(input_data, input_data.t()) + 1e-05 * torch.eye(
        input_data.shape[0]
    )  # make symmetric positive definite
    u = torch.linalg.cholesky(inp)  # pylint: disable=C0103
    output = m(u)
    output_musa = m(u.musa())
    assert testing.DefaultComparator(abs_diff=1e-5)(output, output_musa)


@pytest.mark.parametrize(
    "input_data",
    [
        {"A": torch.randn(1, 3, 3), "B": torch.randn(1, 3, 5)},
        {"A": torch.randn(32, 32, 32), "B": torch.randn(32, 32, 16)},
        {"A": torch.randn(5, 9, 9), "B": torch.randn(5, 9, 7)},
        {"A": torch.randn(3, 3), "B": torch.randn(3, 9)},
        # This size would fail!
        # {"A": torch.randn(128, 128, 128), "B": torch.randn(128, 128, 64)},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_linalg_solve(input_data, dtype):
    input_data["A"] = input_data["A"].to(dtype)
    input_data["B"] = input_data["B"].to(dtype)
    m = torch.linalg.solve
    output = m(input_data["A"].clone(), input_data["B"].clone())
    output_musa = m(
        input_data["A"].to("musa").clone(), input_data["B"].to("musa").clone()
    )
    assert testing.DefaultComparator(abs_diff=1e-4, rel_diff=1e-3)(output, output_musa)


@pytest.mark.parametrize(
    "input_data",
    [
        {"A": torch.randn(1, 3, 3)},
        {"A": torch.randn(16, 16, 16)},
        {"A": torch.randn(5, 9, 9)},
        {"A": torch.randn(3, 3)},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_linalg_lu_factor(input_data, dtype):
    input_data["A"] = input_data["A"].to(dtype)
    m = torch.linalg.lu_factor
    lu, pivot = m(input_data["A"].clone())  # pylint: disable=invalid-name
    lu_musa, pivot_musa = m(input_data["A"].to("musa").clone())
    assert testing.DefaultComparator(abs_diff=1e-5)(lu, lu_musa)
    assert testing.DefaultComparator(abs_diff=1e-5)(pivot, pivot_musa)


@pytest.mark.parametrize(
    "input_data",
    [
        {"A": torch.randn(1, 3, 3)},
        {"A": torch.randn(128, 128)},
        {"A": torch.randn(5, 9, 9)},
        {"A": torch.randn(3, 3)},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_linalg_det(input_data, dtype):
    input_data["A"] = input_data["A"].to(dtype)
    m = torch.linalg.det
    output = m(input_data["A"].clone())
    output_musa = m(input_data["A"].to("musa").clone())
    assert testing.DefaultComparator(abs_diff=1e-5)(output, output_musa)


@pytest.mark.parametrize(
    "input_data",
    [
        {"A": [3, 3], "B": [3, 2]},
        {"A": [3, 3], "B": [3, 3, 2]},
        {"A": [3, 3], "B": [3, 5, 3]},
        {"A": [3, 3], "B": [3, 3, 4]},
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_linalg_lu_solve(input_data, dtype):
    A = torch.randn(*input_data["A"]).to(dtype).musa()
    B = torch.randn(*input_data["B"]).to(dtype).musa()
    LU, pivots = torch.linalg.lu_factor(A)

    if input_data["B"] == [3, 5, 3]:
        X = torch.linalg.lu_solve(LU, pivots, B, left=False)
        res = X @ A
    elif input_data["B"] == [3, 3, 4]:
        X = torch.linalg.lu_solve(LU, pivots, B, adjoint=True)
        res = A.mT @ X
    else:
        X = torch.linalg.lu_solve(LU, pivots, B)
        res = A @ X

    assert testing.DefaultComparator(abs_diff=1e-5)(res, B)
