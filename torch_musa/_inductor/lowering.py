"""MUSA-specific Inductor lowering patches."""

import functools

import sympy
from torch._inductor import lowering
from torch._inductor.virtualized import V


def _statically_known_greater_equal(lhs, rhs):
    return V.graph.sizevars.statically_known_true(sympy.Ge(lhs, rhs))


def _statically_known_equal(lhs, rhs):
    return V.graph.sizevars.statically_known_true(sympy.Eq(lhs, rhs))


def _canonicalize_avg_poolnd_args(x, kernel_size, stride, padding, ceil_mode, dim):
    if x.get_device() is None or x.get_device().type != "musa":
        return None
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0] * dim

    kernel_size = lowering.pad_listlike(kernel_size, dim)
    stride = lowering.pad_listlike(stride, dim)
    padding = lowering.pad_listlike(padding, dim)
    if any(padding):
        return None

    input_size = x.get_size()[-dim:]
    output_size = [
        lowering.pooling_size(
            input_size[i], i, kernel_size, stride, padding, ceil_mode
        )[0]
        for i in range(dim)
    ]
    if not all(
        _statically_known_greater_equal(kernel_size[i], input_size[i])
        and _statically_known_equal(output_size[i], 1)
        for i in range(dim)
    ):
        return None

    # With no explicit padding and one output window, the oversized pooling
    # window contains exactly the input. Upstream lowering then computes the
    # correct in-bounds divisor from these canonical arguments.
    return [
        int(size) if isinstance(size, sympy.Integer) else size for size in input_size
    ]


def _apply_avg_pool_patch():
    original_avg_poolnd = lowering._avg_poolnd

    @functools.wraps(original_avg_poolnd)
    def avg_poolnd(
        x,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dim,
    ):
        canonical_size = _canonicalize_avg_poolnd_args(
            x, kernel_size, stride, padding, ceil_mode, dim
        )
        if canonical_size is not None:
            kernel_size = canonical_size
            stride = canonical_size
        return original_avg_poolnd(
            x,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
            dim,
        )

    lowering._avg_poolnd = avg_poolnd


def _wrap_avg_poolnd_backward(original_backward, dim):
    @functools.wraps(original_backward)
    def avg_poolnd_backward(
        grad_output,
        x,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override=None,
    ):
        canonical_size = _canonicalize_avg_poolnd_args(
            x, kernel_size, stride, padding, ceil_mode, dim
        )
        if canonical_size is not None:
            kernel_size = canonical_size
            stride = canonical_size
        return original_backward(
            grad_output,
            x,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    return avg_poolnd_backward


def _apply_avg_pool_backward_patch():
    for op, dim in (
        (lowering.aten.avg_pool2d_backward.default, 2),
        (lowering.aten.avg_pool3d_backward.default, 3),
    ):
        lowering.lowerings[op] = _wrap_avg_poolnd_backward(lowering.lowerings[op], dim)


def _apply_lowering_patch():
    _apply_avg_pool_patch()
    _apply_avg_pool_backward_patch()


__all__ = ["_apply_lowering_patch"]
