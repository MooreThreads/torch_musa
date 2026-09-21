"""Shared regression helpers for FusedAdam and FusedAdamW tests."""

# pylint: disable=invalid-name,not-callable,use-dict-literal,missing-function-docstring

import copy
import os
import tempfile

import pytest
import torch


class OptimizerTestModel(torch.nn.Module):
    """Small CNN used by the Adam and AdamW training tests."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(256, 120)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(120, 84)
        self.relu4 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(84, 10)
        self.relu5 = torch.nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.reshape(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        return self.relu5(y)


def make_unaligned_params(dtype, device):
    """Create deliberately misaligned views with non-ILP lengths."""
    return [
        torch.nn.Parameter(
            torch.linspace(-1, 1, numel + 1, dtype=dtype, device=device)[1:]
        )
        for numel in (3, 5, 1023, 1025)
    ]


def large_numel_indexing_test(optimizer_cls, amsgrad=False):
    """Regression test for 64-bit tensor and chunk offset arithmetic."""
    numel = 2**31 + 12_345
    parameter = torch.nn.Parameter(
        torch.ones(numel, dtype=torch.float32, device="musa")
    )
    optimizer = optimizer_cls(
        [parameter],
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=amsgrad,
    )
    try:
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        torch.musa.synchronize()
        indices = (0, numel // 2, numel - 1)
        for index in indices:
            assert parameter[index].item() == pytest.approx(0.99, abs=2e-5)
        state = optimizer.state[parameter]
        for index in indices:
            assert state["exp_avg"][index].item() == pytest.approx(0.1, abs=2e-5)
            assert state["exp_avg_sq"][index].item() == pytest.approx(0.001, abs=2e-5)
            if amsgrad:
                assert state["max_exp_avg_sq"][index].item() == pytest.approx(
                    0.001, abs=2e-5
                )
    finally:
        del optimizer
        del parameter
        torch.musa.empty_cache()


def metadata_flush_test(optimizer_cls):
    """Exercise the UINT16 relative chunk-offset flush boundary."""
    chunk_size = 1024
    # This creates chunks [0, ..., UINT16_MAX], with a non-empty scalar tail
    # after the exact relative-offset flush boundary.
    numel = 65_535 * chunk_size + 3
    parameter = torch.nn.Parameter(
        torch.ones(numel, dtype=torch.float32, device="musa")
    )
    optimizer = optimizer_cls(
        [parameter], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    try:
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        torch.musa.synchronize()
        indices = (
            0,
            chunk_size - 1,
            chunk_size,
            65_535 * chunk_size,
            numel - 1,
        )
        for index in indices:
            assert parameter[index].item() == pytest.approx(0.99, abs=2e-5)
        state = optimizer.state[parameter]
        for index in indices:
            assert state["exp_avg"][index].item() == pytest.approx(0.1, abs=2e-5)
            assert state["exp_avg_sq"][index].item() == pytest.approx(0.001, abs=2e-5)
    finally:
        del optimizer
        del parameter
        torch.musa.empty_cache()


def tail_test(
    optimizer_cls,
    reference_cls,
    dtype,
    weight_decay=0.0,
    amsgrad=False,
):
    """Run scalar/tensor-lr tail shapes for ten steps across all dtypes."""
    tolerance = (
        5e-2 if dtype == torch.bfloat16 else (2e-3 if dtype == torch.float16 else 2e-5)
    )
    for numel, tensor_lr, maximize, unaligned in (
        (1025, False, False, False),  # pure tail in the next chunk
        (1041, True, True, True),  # vector prefix plus tail in one chunk
    ):
        lr_value = 1e-2
        kwargs = dict(
            betas=(0.8, 0.9),
            eps=1e-6,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        if unaligned:
            param_musa = torch.nn.Parameter(
                torch.linspace(-0.5, 0.5, numel + 1, dtype=dtype, device="musa")[1:]
            )
        else:
            param_musa = torch.nn.Parameter(
                torch.linspace(-0.5, 0.5, numel, dtype=dtype, device="musa")
            )
        param_ref = torch.nn.Parameter(param_musa.detach().cpu().clone())
        grad_musa = torch.linspace(0.1, 0.3, numel, dtype=dtype, device="musa")
        grad_ref = grad_musa.cpu()
        lr_musa = torch.tensor(lr_value, device="musa") if tensor_lr else lr_value
        optim_musa = optimizer_cls([param_musa], lr=lr_musa, **kwargs)
        optim_ref = reference_cls([param_ref], lr=lr_value, **kwargs)
        for _ in range(10):
            param_musa.grad = grad_musa
            param_ref.grad = grad_ref
            optim_musa.step()
            optim_ref.step()
        torch.musa.synchronize()
        torch.testing.assert_close(
            param_musa.cpu(), param_ref, atol=tolerance, rtol=tolerance
        )
        state_musa = optim_musa.state[param_musa]
        state_ref = optim_ref.state[param_ref]
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                state_musa[name].cpu(),
                state_ref[name],
                atol=tolerance,
                rtol=tolerance,
            )
        if amsgrad:
            torch.testing.assert_close(
                state_musa["max_exp_avg_sq"].cpu(),
                state_ref["max_exp_avg_sq"],
                atol=tolerance,
                rtol=tolerance,
            )


def metadata_flush_tail_after_reset_test(optimizer_cls, reference_cls, weight_decay):
    """Exercise a short tail in the chunk immediately after UINT16 flush."""
    chunk_size = 1024
    # The first short chunk is deliberately after relative offset 65535:
    # chunk 65535 triggers the metadata flush and chunk 65536 contains only
    # three elements.  This is distinct from metadata_flush_test(), where the
    # short chunk is itself the flush-triggering chunk.
    numel = 65_536 * chunk_size + 3
    parameter = torch.nn.Parameter(
        torch.ones(numel, dtype=torch.float32, device="musa")
    )
    optimizer = optimizer_cls(
        [parameter], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay
    )
    reference = torch.nn.Parameter(torch.ones(1))
    reference_optimizer = reference_cls(
        [reference],
        lr=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    try:
        parameter.grad = torch.ones_like(parameter)
        reference.grad = torch.ones_like(reference)
        for _ in range(10):
            optimizer.step()
            reference_optimizer.step()
        torch.musa.synchronize()
        expected = reference.item()
        indices = (
            0,
            chunk_size - 1,
            65_535 * chunk_size,
            65_536 * chunk_size,
            numel - 1,
        )
        for index in indices:
            assert parameter[index].item() == pytest.approx(expected, abs=2e-5)
    finally:
        del optimizer
        del reference_optimizer
        del parameter
        del reference
        torch.musa.empty_cache()


def tail_guard_test(optimizer_cls, reference_cls, dtype, weight_decay):
    """Ensure a partial final vector does not overwrite a trailing sentinel."""
    numel = 1023
    guard_numel = 32
    base = torch.empty(numel + guard_numel, dtype=dtype, device="musa")
    parameter = torch.nn.Parameter(base.narrow(0, 0, numel))
    parameter.data.fill_(1)
    guard = base.narrow(0, numel, guard_numel)
    guard.fill_(17)
    parameter_ref = torch.nn.Parameter(torch.ones(numel, dtype=dtype))
    optimizer = optimizer_cls([parameter], lr=1e-2, weight_decay=weight_decay)
    optimizer_ref = reference_cls([parameter_ref], lr=1e-2, weight_decay=weight_decay)
    parameter.grad = torch.ones_like(parameter)
    parameter_ref.grad = torch.ones_like(parameter_ref)
    for _ in range(10):
        optimizer.step()
        optimizer_ref.step()
    torch.musa.synchronize()
    torch.testing.assert_close(parameter.cpu(), parameter_ref, atol=2e-2, rtol=2e-2)
    assert torch.all(guard == 17).item()


def tensor_lr_dtype_test(optimizer_cls, reference_cls, lr_dtype):
    """Check that non-fp32 tensor learning rates are not silently reinterpreted."""
    numel = 17
    parameter = torch.nn.Parameter(
        torch.ones(numel, dtype=torch.float32, device="musa")
    )
    parameter_ref = torch.nn.Parameter(torch.ones(numel, dtype=torch.float32))
    lr_value = 1e-2
    optimizer = optimizer_cls(
        [parameter], lr=torch.tensor(lr_value, dtype=lr_dtype, device="musa")
    )
    optimizer_ref = reference_cls([parameter_ref], lr=lr_value)
    parameter.grad = torch.ones_like(parameter)
    parameter_ref.grad = torch.ones_like(parameter_ref)
    try:
        optimizer.step()
    except (RuntimeError, ValueError):
        # Explicitly rejecting an unsupported tensor-lr dtype is acceptable;
        # silently treating its storage as float32 is not.
        return
    optimizer_ref.step()
    torch.musa.synchronize()
    torch.testing.assert_close(parameter.cpu(), parameter_ref, atol=2e-5, rtol=2e-5)


def found_inf_test(optimizer_cls, found_inf):
    """Verify found_inf suppresses the update and rolls back the step."""
    param = torch.nn.Parameter(torch.ones(17, device="musa"))
    optimizer = optimizer_cls([param], lr=1e-2)
    optimizer.grad_scale = torch.tensor(2.0, device="musa")
    optimizer.found_inf = torch.tensor(found_inf, device="musa")
    param.grad = torch.ones_like(param)
    optimizer.step()
    torch.musa.synchronize()
    if found_inf:
        assert optimizer.state[param]["step"].item() == pytest.approx(0.0)
    else:
        assert optimizer.state[param]["step"].item() == pytest.approx(1.0)


def empty_and_missing_grad_test(optimizer_cls, weight_decay, expected_active):
    """Empty tensors and parameters without gradients must be skipped."""
    empty = torch.nn.Parameter(torch.empty(0, device="musa"))
    active = torch.nn.Parameter(torch.ones(17, device="musa"))
    no_grad = torch.nn.Parameter(torch.ones(17, device="musa"))
    optimizer = optimizer_cls(
        [empty, active, no_grad], lr=1e-2, weight_decay=weight_decay
    )
    active.grad = torch.ones_like(active)
    optimizer.step()
    torch.musa.synchronize()
    assert active[0].item() == pytest.approx(expected_active, abs=2e-5)
    assert no_grad[0].item() == pytest.approx(1.0)
    assert empty not in optimizer.state
    assert no_grad not in optimizer.state


def basic_test(optimizer_cls, reference_cls, weight_decay, amsgrad, dtype):
    """Compare several standard optimizer steps on a large tensor."""
    t = torch.zeros(1024 * 1024 * 8, dtype=dtype, device="musa")
    t_ref = torch.zeros(1024 * 1024 * 8, dtype=dtype, device="cpu")
    lr = 0.002
    optimizer_fused = optimizer_cls(
        [t], lr=lr, weight_decay=weight_decay, amsgrad=amsgrad
    )
    optimizer_ref = reference_cls(
        [t_ref], lr=lr, weight_decay=weight_decay, amsgrad=amsgrad
    )

    for _ in range(3):
        optimizer_fused.zero_grad()
        optimizer_ref.zero_grad()
        grad = torch.randn_like(t)
        t.grad = grad
        t_ref.grad = grad.cpu()
        optimizer_fused.step()
        optimizer_ref.step()
        torch.testing.assert_close(t.cpu(), t_ref.cpu())
        torch.musa.synchronize()


def unaligned_test(optimizer_cls, reference_cls, dtype, amsgrad, weight_decay):
    """Compare misaligned pointers, ILP tails and partial chunks."""
    params_musa = make_unaligned_params(dtype, "musa")
    # Keep the golden optimizer in FP32 so small coupled-decay gradients do
    # not underflow in the reference moments for low-precision parameters.
    params_ref = [
        torch.nn.Parameter(param.detach().float().clone()) for param in params_musa
    ]
    ilp = 16 if dtype == torch.float32 else 4
    assert all(param.numel() % ilp for param in params_musa)
    assert all(param.data_ptr() % (ilp * param.element_size()) for param in params_musa)

    kwargs = dict(
        lr=1e-2,
        betas=(0.8, 0.9),
        eps=1e-6,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )
    optim_ref = reference_cls(params_ref, **kwargs)
    optim_musa = optimizer_cls(params_musa, **kwargs)
    for step in range(3):
        for param_ref, param_musa in zip(params_ref, params_musa):
            grad = torch.linspace(
                0.1 * (step + 1),
                0.2 * (step + 1),
                param_ref.numel(),
                dtype=dtype,
                device="musa",
            )
            param_ref.grad = grad.float()
            param_musa.grad = grad
        optim_ref.step()
        optim_musa.step()

    tolerance = 2e-2 if dtype == torch.bfloat16 else 5e-3
    for param_ref, param_musa in zip(params_ref, params_musa):
        torch.testing.assert_close(
            param_musa.float(), param_ref, atol=tolerance, rtol=tolerance
        )
        state_ref, state_musa = optim_ref.state[param_ref], optim_musa.state[param_musa]
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                state_musa[name].float(),
                state_ref[name],
                atol=tolerance,
                rtol=tolerance,
            )
        if amsgrad:
            torch.testing.assert_close(
                state_musa["max_exp_avg_sq"].float(),
                state_ref["max_exp_avg_sq"],
                atol=tolerance,
                rtol=tolerance,
            )


def training_test(optimizer_cls, reference_cls):
    """Compare a short fp32 CNN training run against the torch optimizer."""
    lr = 0.001
    model = OptimizerTestModel().to("musa")
    model_ref = OptimizerTestModel().to("musa")
    model.load_state_dict(copy.deepcopy(model_ref.state_dict()))
    params = [p for p in model.parameters() if p.requires_grad]
    params_ref = [p for p in model_ref.parameters() if p.requires_grad]
    optimizer = optimizer_cls(params, lr=lr)
    optimizer_ref = reference_cls(params_ref, lr=lr)

    for _ in range(20):
        x = torch.rand([32, 1, 28, 28], device="musa")
        x_ref = x.clone()
        gt = torch.rand([32, 10], device="musa")
        gt_ref = gt.clone()
        loss = ((gt - model(x)) ** 2).mean()
        loss.backward()
        optimizer.step()
        loss_ref = ((gt_ref - model_ref(x_ref)) ** 2).mean()
        loss_ref.backward()
        optimizer_ref.step()

        for m, m_ref in zip(model.modules(), model_ref.modules()):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.testing.assert_close(
                    m.weight, m_ref.weight, atol=1e-5, rtol=1e-5, equal_nan=True
                )
                torch.testing.assert_close(
                    m.weight.grad,
                    m_ref.weight.grad,
                    atol=1e-5,
                    rtol=1e-5,
                    equal_nan=True,
                )

        for i, state in enumerate(optimizer.state.values()):
            state["step"] += i
        for i, state in enumerate(optimizer_ref.state.values()):
            state["step"] += i
        optimizer.zero_grad()
        optimizer_ref.zero_grad()
        model.load_state_dict(copy.deepcopy(model_ref.state_dict()))


def amp_training_test(optimizer_cls, reference_cls):
    """Compare mixed-precision CNN training against the torch optimizer."""
    lr = 1e-4
    model = OptimizerTestModel().to("musa")
    model_ref = OptimizerTestModel().to("musa")
    model.load_state_dict(copy.deepcopy(model_ref.state_dict()))
    scaler = torch.musa.amp.GradScaler(enabled=True)
    scaler_ref = torch.musa.amp.GradScaler(enabled=True)
    params = [p for p in model.parameters() if p.requires_grad]
    params_ref = [p for p in model_ref.parameters() if p.requires_grad]
    optimizer = optimizer_cls(params, lr=lr)
    optimizer_ref = reference_cls(params_ref, lr=lr)

    for _ in range(20):
        x = torch.rand([32, 1, 28, 28], device="musa")
        x_ref = x.clone()
        gt = torch.rand([32, 10], device="musa")
        gt_ref = gt.clone()
        with torch.musa.amp.autocast(enabled=True):
            loss = ((gt - model(x)) ** 2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.musa.amp.autocast(enabled=True):
            loss_ref = ((gt_ref - model_ref(x_ref)) ** 2).mean()
        scaler_ref.scale(loss_ref).backward()
        scaler_ref.step(optimizer_ref)
        scaler_ref.update()

        for m, m_ref in zip(model.modules(), model_ref.modules()):
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.testing.assert_close(
                    m.weight, m_ref.weight, atol=1e-3, rtol=1e-3, equal_nan=True
                )
                torch.testing.assert_close(
                    m.weight.grad,
                    m_ref.weight.grad,
                    atol=1e-3,
                    rtol=1e-3,
                    equal_nan=True,
                )
        optimizer.zero_grad()
        optimizer_ref.zero_grad()
        model.load_state_dict(copy.deepcopy(model_ref.state_dict()))


def state_dict_test(optimizer_cls):
    """Verify optimizer state serialization keeps host-side step values."""
    model = OptimizerTestModel().to("musa")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optimizer_cls(params, lr=0.001)
    total_steps = 5
    for _ in range(total_steps):
        x = torch.rand([32, 1, 28, 28], device="musa")
        gt = torch.rand([32, 10], device="musa")
        loss = ((gt - model(x)) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = os.path.join(tmp_dir, "state_dict.pth")
        torch.save({"optimizer_state_dict": optimizer.state_dict()}, filename)
        optimizer_state_dict = torch.load(filename, weights_only=True)[
            "optimizer_state_dict"
        ]
        for param_state in optimizer_state_dict["state"].values():
            if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                assert param_state["step"].device.type == "musa"
            else:
                assert param_state["step"].device.type == "cpu"
            assert param_state["step"].item() == total_steps
