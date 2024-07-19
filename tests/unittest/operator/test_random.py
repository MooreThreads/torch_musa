"""Test random operators."""

# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import, invalid-name, too-many-nested-blocks
import torch
import pytest
from torch_musa import testing
import torch_musa

size = [(2,), (2, 3), (2, 3, 4)]
low = [1, 10]
high = [20, 30]

generator = [None, torch.Generator("musa")]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("size", size)
@pytest.mark.parametrize("low", low)
@pytest.mark.parametrize("high", high)
@pytest.mark.parametrize("generator", generator)
def test_randint_low_generator(low, high, size, generator):
    input_params = {
        "low": low,
        "high": high,
        "size": size,
        "generator": generator,
        "device": "musa",
    }
    test = testing.OpTest(
        func=torch.randint, input_args=input_params, ignored_result_indices=[0]
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("size", size)
@pytest.mark.parametrize("low", low)
@pytest.mark.parametrize("high", high)
def test_randint_low(low, high, size):
    input_params = {"low": low, "high": high, "size": size, "device": "musa"}
    test = testing.OpTest(
        func=torch.randint, input_args=input_params, ignored_result_indices=[0]
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("size", size)
@pytest.mark.parametrize("high", high)
def test_randint(high, size):
    input_params = {"high": high, "size": size, "device": "musa"}
    test = testing.OpTest(
        func=torch.randint, input_args=input_params, ignored_result_indices=[0]
    )
    test.check_result()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("size", size)
@pytest.mark.parametrize("high", high)
@pytest.mark.parametrize("generator", generator)
def test_randint_generator(high, size, generator):
    input_params = {
        "high": high,
        "size": size,
        "generator": generator,
        "device": "musa",
    }
    test = testing.OpTest(
        func=torch.randint, input_args=input_params, ignored_result_indices=[0]
    )
    test.check_result()


dtypes = [
    torch.float,
    torch.double,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.half] + dtypes)
def test_random(dtype):
    t = torch.empty(200, dtype=dtype, device="musa")
    lb = 1
    ub = 4

    t.fill_(-1)
    t.random_(lb, ub)
    assert t.cpu().min().item() == lb
    assert t.cpu().max().item() == ub - 1

    t.fill_(-1)
    t.random_(ub)
    assert t.cpu().min().item() == 0
    assert t.cpu().max().item() == ub - 1


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_random_bool():
    size = 2000
    t = torch.empty(size, dtype=torch.bool, device="musa")

    t.fill_(False)
    t.random_()
    assert t.min().item() is False
    assert t.max().item() is True
    assert 0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6

    t.fill_(True)
    t.random_()
    assert t.min().item() is False
    assert t.max().item() is True
    assert 0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6


int64_min_val = torch.iinfo(torch.int64).min
int64_max_val = torch.iinfo(torch.int64).max
min_val = 0
max_val = 1
froms = [int64_min_val, -42, min_val - 1, min_val, max_val, max_val + 1, 42]
tos = [-42, min_val - 1, min_val, max_val, max_val + 1, 42, int64_max_val]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("from_", froms)
@pytest.mark.parametrize("to_", tos)
def test_random_from_to_bool(from_, to_):
    size = 2000

    t = torch.empty(size, dtype=torch.bool, device="musa")
    if to_ > from_:
        if not min_val <= from_ <= max_val:
            with pytest.raises(RuntimeError, match="from is out of bounds"):
                t.random_(from_, to_)
        elif not min_val <= (to_ - 1) <= max_val:
            with pytest.raises(RuntimeError, match="to - 1 is out of bounds"):
                t.random_(from_, to_)
        else:
            t.random_(from_, to_)
            delta = 1
            assert from_ <= t.to(torch.int).min().item() < (from_ + delta)
            assert (to_ - delta) <= t.to(torch.int).max().item() < to_
    else:
        with pytest.raises(
            RuntimeError,
            match="random_ expects 'from' to be less than 'to', but got from="
            + str(from_)
            + " >= to="
            + str(to_),
        ):
            t.random_(from_, to_)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.half] + dtypes)
def test_random_full_range(dtype):
    size = 2000
    alpha = 0.1

    int64_min_val = torch.iinfo(torch.int64).min
    int64_max_val = torch.iinfo(torch.int64).max

    if dtype == torch.double:
        fp_limit = 2**53
    elif dtype == torch.float:
        fp_limit = 2**24
    elif dtype == torch.half:
        fp_limit = 2**11
    else:
        fp_limit = 0

    t = torch.empty(size, dtype=dtype, device="musa")

    if dtype in [torch.float, torch.double, torch.half]:
        from_ = int(max(-fp_limit, int64_min_val))
        to_inc_ = int(min(fp_limit, int64_max_val))
    else:
        from_ = int(max(torch.iinfo(dtype).min, int64_min_val))
        to_inc_ = int(min(torch.iinfo(dtype).max, int64_max_val))
    range_ = to_inc_ - from_ + 1

    t.random_(from_, None)
    delta = max(1, alpha * range_)
    assert from_ <= t.cpu().to(torch.double).min().item() < (from_ + delta)
    assert (to_inc_ - delta) < t.cpu().to(torch.double).max().item() <= to_inc_


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.half] + dtypes)
def test_random_from_to(dtype):
    size = 2000
    alpha = 0.1

    int64_min_val = torch.iinfo(torch.int64).min
    int64_max_val = torch.iinfo(torch.int64).max

    if dtype in [torch.float, torch.double, torch.half]:
        min_val = int(max(torch.finfo(dtype).min, int64_min_val))
        max_val = int(min(torch.finfo(dtype).max, int64_max_val))
        froms = [min_val, -42, 0, 42]
        tos = [-42, 0, 42, max_val >> 1]
    elif dtype == torch.uint8:
        min_val = torch.iinfo(dtype).min
        max_val = torch.iinfo(dtype).max
        froms = [int64_min_val, -42, min_val - 1, min_val, 42, max_val, max_val + 1]
        tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
    elif dtype == torch.int64:
        min_val = int64_min_val
        max_val = int64_max_val
        froms = [min_val, -42, 0, 42]
        tos = [-42, 0, 42, max_val]
    else:
        min_val = torch.iinfo(dtype).min
        max_val = torch.iinfo(dtype).max
        froms = [int64_min_val, min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1]
        tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

    if dtype == torch.double:
        fp_limit = 2**53
    elif dtype == torch.float:
        fp_limit = 2**24
    elif dtype == torch.half:
        fp_limit = 2**11
    else:
        fp_limit = 0

    for from_ in froms:
        for to_ in tos:
            t = torch.empty(size, dtype=dtype, device="musa")
            if to_ > from_:
                if not min_val <= from_ <= max_val:
                    with pytest.raises(RuntimeError, match="from is out of bounds"):
                        t.random_(from_, to_)
                elif not min_val <= (to_ - 1) <= max_val:
                    with pytest.raises(RuntimeError, match="to - 1 is out of bounds"):
                        t.random_(from_, to_)
                else:
                    if dtype.is_floating_point and (
                        not (-fp_limit <= from_ <= fp_limit)
                        or not (-fp_limit <= (to_ - 1) <= fp_limit)
                    ):
                        if not -fp_limit <= from_ <= fp_limit:
                            with pytest.warns(
                                UserWarning, match="from is out of bounds"
                            ):
                                t.random_(from_, to_)
                        if not -fp_limit <= (to_ - 1) <= fp_limit:
                            with pytest.warns(
                                UserWarning, match="to - 1 is out of bounds"
                            ):
                                t.random_(from_, to_)
                    else:
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = max(1, alpha * range_)
                        assert (
                            from_
                            <= t.cpu().to(torch.double).min().item()
                            < (from_ + delta)
                        )
                        assert (
                            (to_ - delta) <= t.cpu().to(torch.double).max().item() < to_
                        )
            else:
                with pytest.raises(
                    RuntimeError,
                    match="random_ expects 'from' to be less than 'to', but got from="
                    + str(from_)
                    + " >= to="
                    + str(to_),
                ):
                    t.random_(from_, to_)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.half] + dtypes)
def test_random_to(dtype):
    size = 2000
    alpha = 0.1

    int64_min_val = torch.iinfo(torch.int64).min
    int64_max_val = torch.iinfo(torch.int64).max

    if dtype in [torch.float, torch.double, torch.half]:
        min_val = int(max(torch.finfo(dtype).min, int64_min_val))
        max_val = int(min(torch.finfo(dtype).max, int64_max_val))
        tos = [-42, 0, 42, max_val >> 1]
    elif dtype == torch.uint8:
        min_val = torch.iinfo(dtype).min
        max_val = torch.iinfo(dtype).max
        tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
    elif dtype == torch.int64:
        min_val = int64_min_val
        max_val = int64_max_val
        tos = [-42, 0, 42, max_val]
    else:
        min_val = torch.iinfo(dtype).min
        max_val = torch.iinfo(dtype).max
        tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

    from_ = 0
    for to_ in tos:
        t = torch.empty(size, dtype=dtype, device="musa")
        if to_ > from_:
            if not min_val <= (to_ - 1) <= max_val:
                with pytest.raises(RuntimeError, match="to - 1 is out of bounds"):
                    t.random_(from_, to_)
            else:
                t.random_(to_)
                range_ = to_ - from_
                delta = max(1, alpha * range_)
                # Don't check this warning
                assert from_ <= t.cpu().to(torch.double).min().item() < (from_ + delta)
                assert (to_ - delta) <= t.cpu().to(torch.double).max().item() < to_
        else:
            with pytest.raises(
                RuntimeError,
                match="random_ expects 'from' to be less than 'to', but got from="
                + str(from_)
                + " >= to="
                + str(to_),
            ):
                t.random_(from_, to_)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.half] + dtypes)
def test_random_default(dtype):
    size = 2000
    alpha = 0.1

    if dtype == torch.float:
        to_inc = 1 << 24
    elif dtype == torch.double:
        to_inc = 1 << 53
    elif dtype == torch.half:
        to_inc = 1 << 11
    else:
        to_inc = torch.iinfo(dtype).max

    t = torch.empty(size, dtype=dtype, device="musa")
    t.random_()
    assert 0 <= t.cpu().to(torch.double).min().item() < alpha * to_inc
    assert (to_inc - alpha * to_inc) < t.cpu().to(torch.double).max().item() <= to_inc


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "dtype", [torch.double, torch.float, torch.long, torch.int, torch.short]
)
def test_random_neg_values(dtype):
    SIZE = 10
    res = torch.rand(SIZE, SIZE).to(device="musa", dtype=dtype)
    res.random_(-10, -1)
    assert res.cpu().max().item() <= -1
    assert res.cpu().min().item() >= -10
