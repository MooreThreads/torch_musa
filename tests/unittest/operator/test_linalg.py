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
    assert testing.DefaultComparator(abs_diff=2e-5)(output, output_musa)


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


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "input_data",
    [
        {"A": torch.randn([7, 7], dtype=torch.complex64), "mode": "reduced"},
        {"A": torch.randn([25, 18], dtype=torch.complex64), "mode": "reduced"},
        {"A": torch.randn([18, 9]), "mode": "reduced"},
        {"A": torch.randn([25, 18]), "mode": "reduced"},
    ],
)
def test_qr(input_data):
    test = testing.OpTest(
        func=torch.linalg.qr,
        input_args=input_data,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


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


MUSA_DEVICE_DECORATOR = testing.test_on_nonzero_card_if_multiple_musa_device(1)


@pytest.mark.parametrize(
    "input_config",
    [
        {
            "A_shape": [3, 3],
            "B_shape": [3, 2],
            "upper": False,
            "transpose": False,
            "unitriangular": False,
        },
        {
            "A_shape": [4, 4],
            "B_shape": [4, 1],
            "upper": True,
            "transpose": False,
            "unitriangular": False,
        },
        {
            "A_shape": [2, 2],
            "B_shape": [2, 3],
            "upper": False,
            "transpose": True,
            "unitriangular": False,
        },
        {
            "A_shape": [5, 5],
            "B_shape": [5, 2],
            "upper": True,
            "transpose": True,
            "unitriangular": False,
        },
        {
            "A_shape": [3, 3],
            "B_shape": [3, 2],
            "upper": False,
            "transpose": False,
            "unitriangular": True,
        },
        {
            "A_shape": [4, 4],
            "B_shape": [4, 1],
            "upper": True,
            "transpose": False,
            "unitriangular": True,
        },
        {
            "A_shape": [2, 2],
            "B_shape": [2, 3],
            "upper": False,
            "transpose": True,
            "unitriangular": True,
        },
        {
            "A_shape": [5, 5],
            "B_shape": [5, 2],
            "upper": True,
            "transpose": True,
            "unitriangular": True,
        },
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@MUSA_DEVICE_DECORATOR
def test_triangular_solve(input_config, dtype):
    A_shape = input_config["A_shape"]
    B_shape = input_config["B_shape"]
    upper = input_config["upper"]
    transpose = input_config["transpose"]
    unitriangular = input_config["unitriangular"]

    A_full = torch.randn(*A_shape, dtype=dtype)
    B = torch.randn(*B_shape, dtype=dtype)

    if upper:
        A_for_solve = torch.triu(A_full)
    else:
        A_for_solve = torch.tril(A_full)

    diag_indices = torch.arange(A_shape[-1])
    if not unitriangular:
        A_for_solve[..., diag_indices, diag_indices] = (
            A_for_solve[..., diag_indices, diag_indices] + 1.0
        )

    A_for_verification = A_for_solve.clone()
    if unitriangular:
        A_for_verification[..., diag_indices, diag_indices] = 1.0

    A_for_solve = A_for_solve.musa()
    B = B.musa()
    A_for_verification = A_for_verification.musa()

    X, _ = torch.triangular_solve(
        B, A_for_solve, upper=upper, transpose=transpose, unitriangular=unitriangular
    )

    if transpose:
        res = A_for_verification.mT @ X
    else:
        res = A_for_verification @ X

    # todo print X mudnn fail
    assert testing.DefaultComparator(abs_diff=1e-5)(res, B)


@pytest.mark.parametrize(
    "A_shape",
    [[3, 3], [5, 5], [4, 3], [6, 4], [3, 4], [4, 6], [2, 3, 3], [3, 5, 5]],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@MUSA_DEVICE_DECORATOR
def test_linalg_lu(A_shape, dtype):
    A = torch.randn(*A_shape, dtype=dtype)

    inputs = {"A": A}
    test = testing.OpTest(
        func=torch.linalg.lu,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )

    test.check_result()
    test.check_out_ops()


@pytest.mark.parametrize(
    "A_shape",
    [
        [3, 3],
        [5, 5],
        [4, 3],
        [3, 4],
        [2, 3, 3],
        [2, 4, 3],
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@MUSA_DEVICE_DECORATOR
def test_linalg_lu_out(A_shape, dtype):
    A = torch.randn(*A_shape, dtype=dtype)

    if len(A_shape) == 2:
        P_out = torch.empty(A_shape[0], A_shape[0], dtype=dtype, device="musa")
        L_out = torch.empty(A_shape[0], A_shape[0], dtype=dtype, device="musa")
        U_out = torch.empty(A_shape[0], A_shape[1], dtype=dtype, device="musa")
    else:
        batch_size, m, n = A_shape
        P_out = torch.empty(batch_size, m, m, dtype=dtype, device="musa")
        L_out = torch.empty(batch_size, m, m, dtype=dtype, device="musa")
        U_out = torch.empty(batch_size, m, n, dtype=dtype, device="musa")

    torch.linalg.lu(A.to("musa"), out=(P_out, L_out, U_out))

    result = P_out @ L_out @ U_out

    assert testing.DefaultComparator(abs_diff=1e-2)(A.to("musa"), result)

    if len(A_shape) == 2:
        assert torch.allclose(L_out.tril(), L_out, rtol=1e-4, atol=1e-6)
        assert torch.allclose(
            L_out.diag(), torch.ones_like(L_out.diag()), rtol=1e-4, atol=1e-6
        )
        assert torch.allclose(U_out.triu(), U_out, rtol=1e-4, atol=1e-6)
        assert torch.allclose(
            P_out @ P_out.T, torch.eye(A_shape[0], device="musa"), rtol=1e-4, atol=1e-6
        )
    else:
        batch_size = A_shape[0]
        for i in range(batch_size):
            assert torch.allclose(L_out[i].tril(), L_out[i], rtol=1e-4, atol=1e-6)
            assert torch.allclose(
                L_out[i].diag(), torch.ones_like(L_out[i].diag()), rtol=1e-4, atol=1e-6
            )
            assert torch.allclose(U_out[i].triu(), U_out[i], rtol=1e-4, atol=1e-6)
            assert torch.allclose(
                P_out[i] @ P_out[i].T,
                torch.eye(A_shape[1], device="musa"),
                rtol=1e-4,
                atol=1e-6,
            )


@pytest.mark.parametrize(
    "A_shape",
    [[3, 3], [5, 5], [4, 4], [6, 6], [2, 3, 3], [3, 5, 5]],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@MUSA_DEVICE_DECORATOR
def test_linalg_ldl_factor_ex(A_shape, dtype):

    if len(A_shape) == 2:
        m, _ = A_shape
        A = torch.randn(m, m, dtype=dtype)
        A = A @ A.T + torch.eye(m, dtype=dtype) * 0.1
    else:
        batch, m, _ = A_shape
        A = torch.randn(batch, m, m, dtype=dtype)
        A = A @ A.transpose(1, 2) + torch.eye(m, dtype=dtype).unsqueeze(0) * 0.1

    inputs = {"input": A, "hermitian": True}
    test = testing.OpTest(
        func=torch.linalg.ldl_factor_ex,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )

    test.check_result()
    test.check_out_ops()

    if len(A_shape) == 2:
        m, _ = A_shape
        A = torch.randn(m, m, dtype=dtype)
    else:
        batch, m, _ = A_shape
        A = torch.randn(batch, m, m, dtype=dtype)

    inputs = {"input": A, "hermitian": False}
    test = testing.OpTest(
        func=torch.linalg.ldl_factor_ex,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )

    test.check_result()
    test.check_out_ops()


@pytest.mark.parametrize(
    "A_shape",
    [[3, 3], [5, 5], [4, 4], [6, 6], [2, 3, 3], [3, 5, 5]],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@MUSA_DEVICE_DECORATOR
def test_linalg_ldl_solve(A_shape, dtype):

    if len(A_shape) == 2:
        m, _ = A_shape
        A = torch.randn(m, m, dtype=dtype)
        A = A @ A.T + torch.eye(m, dtype=dtype) * 0.1
        B = torch.randn(m, 2, dtype=dtype)
    else:
        batch, m, _ = A_shape
        A = torch.randn(batch, m, m, dtype=dtype)
        A = A @ A.transpose(1, 2) + torch.eye(m, dtype=dtype).unsqueeze(0) * 0.1
        B = torch.randn(batch, m, 2, dtype=dtype)

    LD, pivots, _ = torch.linalg.ldl_factor_ex(A, hermitian=True)

    inputs = {"LD": LD, "pivots": pivots, "B": B, "hermitian": True}
    test = testing.OpTest(
        func=torch.linalg.ldl_solve,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )

    test.check_result()
    test.check_out_ops()

    out = torch.empty_like(B)
    inputs = {"LD": LD, "pivots": pivots, "B": B, "hermitian": True, "out": out}
    test = testing.OpTest(
        func=torch.linalg.ldl_solve,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )

    test.check_result()
    test.check_out_ops()


@pytest.mark.parametrize(
    "A_shape",
    [[3, 3], [5, 5], [4, 4], [6, 6], [2, 3, 3], [3, 5, 5]],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@MUSA_DEVICE_DECORATOR
def test_linalg_slogdet(A_shape, dtype):

    if len(A_shape) == 2:
        m, _ = A_shape
        A = torch.randn(m, m, dtype=dtype)
    else:
        batch, m, _ = A_shape
        A = torch.randn(batch, m, m, dtype=dtype)

    inputs = {"A": A}
    test = testing.OpTest(
        func=torch._linalg_slogdet,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )

    test.check_result()
    test.check_out_ops()


@pytest.mark.parametrize(
    "A_shape", [[3, 3], [5, 5], [4, 4], [6, 6], [2, 3, 3], [3, 5, 5]]
)
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("compute_v", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
@MUSA_DEVICE_DECORATOR
def test_linalg_eigh(A_shape, uplo, compute_v, dtype):

    if len(A_shape) == 2:
        m, _ = A_shape
        A = torch.randn(m, m, dtype=dtype)
        A = A @ A.T
    else:
        batch, m, _ = A_shape
        A = torch.randn(batch, m, m, dtype=dtype)
        A = A @ A.transpose(1, 2)

    inputs = {"A": A, "UPLO": uplo, "compute_v": compute_v}
    test = testing.OpTest(
        func=torch._linalg_eigh,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )

    test.check_result()
    test.check_out_ops()


@pytest.mark.parametrize(
    "A_shape",
    [[3, 3], [5, 5], [4, 6], [6, 4], [2, 3, 3], [3, 4, 5]],  # 包含非方阵和批量
)
@pytest.mark.parametrize("full_matrices", [True, False])
@pytest.mark.parametrize("compute_uv", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
@MUSA_DEVICE_DECORATOR
def test_linalg_svd(A_shape, full_matrices, compute_uv, dtype):

    if len(A_shape) == 2:
        m, n = A_shape
        A = torch.randn(m, n, dtype=dtype)
    else:
        batch, m, n = A_shape
        A = torch.randn(batch, m, n, dtype=dtype)

    inputs = {"A": A, "full_matrices": full_matrices, "compute_uv": compute_uv}
    test = testing.OpTest(
        func=torch._linalg_svd,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )

    test.check_result()
    test.check_out_ops()


@pytest.mark.parametrize(
    "A_shape",
    [
        [3, 3],
        [5, 5],
        [2, 3, 3],
        [4, 4, 4],
        [4, 2, 1],  # batch
    ],
)
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cholesky_solve_helper(A_shape, upper, dtype):
    if len(A_shape) == 2:
        n, _ = A_shape
        M = torch.randn(n, n, dtype=dtype)
        A = M @ M.T + 1e-3 * torch.eye(n, dtype=dtype)
        B = torch.randn(n, 2, dtype=dtype)
    else:
        batch, n, _ = A_shape
        M = torch.randn(batch, n, n, dtype=dtype)
        A = M @ M.mT + 1e-3 * torch.eye(n, dtype=dtype)
        B = torch.randn(batch, n, 2, dtype=dtype)

    chol = torch.linalg.cholesky(A, upper=upper)

    inputs = {
        "input": B,
        "input2": chol,
        "upper": upper,
    }

    test = testing.OpTest(
        func=torch.cholesky_solve,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )

    test.check_result()
    test.check_out_ops()


@pytest.mark.parametrize(
    "A_shape",
    [
        [4, 4],
        [6, 4],  # M > N
        [3, 5],  # M < N
        [2, 4, 4],  # Batch
        [3, 5, 2],  # Batch M > N
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_geqrf(A_shape, dtype):
    """
    UT for torch.geqrf
    """
    A = torch.randn(*A_shape, dtype=dtype)
    A_musa = A.musa()

    res_a_cpu, res_tau_cpu = torch.geqrf(A)
    res_a_musa, res_tau_musa = torch.geqrf(A_musa)

    torch.allclose(res_a_cpu, res_a_musa.cpu(), rtol=1e-5, atol=1e-5)
    torch.allclose(res_tau_cpu, res_tau_musa.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "shape",
    [
        (5, 3),  # M=5, N=3
        (3, 3),  # M=3, N=3
        (2, 6, 4),  # Batch M=6, N=4
    ],
)
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ormqr(shape, left, transpose, dtype):
    """
    UT for torch.ormqr
    """
    A = torch.randn(*shape, dtype=dtype)
    a, tau = torch.geqrf(A)

    m, batch = shape[-2], shape[:-2]

    if left:
        B_shape = batch + (m, 2)
    else:
        B_shape = batch + (2, m)

    B = torch.randn(*B_shape, dtype=dtype)

    a_musa = a.musa()
    tau_musa = tau.musa()
    B_musa = B.musa()

    res_cpu = torch.ormqr(a, tau, B, left=left, transpose=transpose)
    res_musa = torch.ormqr(a_musa, tau_musa, B_musa, left=left, transpose=transpose)

    torch.allclose(res_cpu, res_musa.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (5, 5),
        (2, 4, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eig(shape, dtype):
    """
    UT for torch.linalg.eig
    """
    A = torch.randn(*shape, dtype=dtype)
    inputs = {"input": A}
    test = testing.OpTest(
        func=torch.linalg.eig,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    test.check_result()
    test.check_out_ops()


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (5, 5),
        (2, 4, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals(shape, dtype):
    """
    Unit test for torch.linalg.eigvals using `out=` parameter on CPU and MUSA.

    Checks:
        1. Eigvals computed with `out=` match between CPU and MUSA.
    """

    A = torch.randn(*shape, dtype=dtype)

    inputs = {"input": A}
    test = testing.OpTest(
        func=torch.linalg.eigvals,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    test.check_result()
    test.check_out_ops()


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (6, 4),
        (10, 8),
        (2, 5, 3),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_householder_product(shape, dtype):
    """
    UT for torch.linalg.householder_product
    """

    A = torch.randn(*shape, dtype=dtype)
    input_cpu, tau_cpu = torch.geqrf(A)

    inputs = {"input": input_cpu, "tau": tau_cpu}

    test = testing.OpTest(
        func=torch.linalg.householder_product,
        input_args=inputs,
        comparators=testing.DefaultComparator(abs_diff=1e-5),
    )
    test.check_result()
    test.check_out_ops()


# TODO: torch_musa already implements linalg_solve_triangular,
#       but it currently has a bug in musolver (musaFree).
