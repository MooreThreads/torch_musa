"""FusedSGD Test"""

# pylint: disable=unused-import,invalid-name,not-callable,missing-function-docstring
import copy
import os
import tempfile

import pytest
import torch
import torch_musa
from torch_musa import testing


# Borrowed pattern from fused optimizer tests (Apex-style small CNN)
class Model(torch.nn.Module):
    """Model for testing fused SGD"""

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
        return y


def _make_sgd(
    param_list,
    lr,
    momentum,
    dampening,
    weight_decay,
    nesterov,
    maximize,
    fused,
    foreach,
):
    # NOTE: torch.optim.SGD supports lr as float or Tensor
    # and supports fused/foreach knobs.
    return torch.optim.SGD(
        param_list,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        maximize=maximize,
        fused=fused,
        foreach=foreach,
    )


class TestFusedSGD:
    """class of fused SGD unit test suits"""

    @testing.test_on_nonzero_card_if_multiple_musa_device(1)
    @pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
    @pytest.mark.parametrize("momentum", [0.0, 0.9])
    @pytest.mark.parametrize("nesterov", [False, True])
    @pytest.mark.parametrize("maximize", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_basic(self, weight_decay, momentum, nesterov, maximize, dtype):
        # nesterov only meaningful when momentum > 0
        if momentum == 0.0 and nesterov:
            pytest.skip("nesterov requires momentum > 0")

        # Keep dampening non-zero in momentum branch to cover that path too,
        # but Nesterov requires dampening == 0.
        if momentum == 0.0:
            dampening = 0.0
        elif nesterov:
            dampening = 0.0
        else:
            dampening = 0.1

        # Use same-device reference to avoid CPU-vs-device numeric drift
        t = torch.zeros(1024 * 1024, dtype=dtype, device="musa")
        t_ref = t.clone()

        lr = 2e-3

        # fused=True path (should call into _fused_sgd_*)
        opt_fused = _make_sgd(
            [t],
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            fused=True,
            foreach=False,  # force not foreach, so fused is the chosen fastpath
        )

        # reference: force for-loop by disabling fused/foreach
        opt_ref = _make_sgd(
            [t_ref],
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            fused=False,
            foreach=False,
        )

        for _ in range(5):
            opt_fused.zero_grad(set_to_none=True)
            opt_ref.zero_grad(set_to_none=True)

            grad = torch.randn_like(t)
            t.grad = grad
            t_ref.grad = grad.clone()

            opt_fused.step()
            opt_ref.step()

            torch.testing.assert_close(t, t_ref, atol=1e-6, rtol=1e-6)
            torch.musa.synchronize()

    @testing.test_on_nonzero_card_if_multiple_musa_device(1)
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_tensor_lr(self, dtype):
        # NOTE: lr as Tensor should be a 0-dim tensor in practice.
        # (1-dim size-1 has been reported to trigger runtime errors in some checks.)
        lr = torch.tensor(2e-3, device="musa", dtype=dtype)

        t = torch.zeros(1024 * 1024, dtype=dtype, device="musa")
        t_ref = t.clone()

        opt_fused = _make_sgd(
            [t],
            lr=lr,
            momentum=0.9,
            dampening=0.0,
            weight_decay=1e-2,
            nesterov=True,
            maximize=False,
            fused=True,
            foreach=False,
        )
        opt_ref = _make_sgd(
            [t_ref],
            lr=lr,
            momentum=0.9,
            dampening=0.0,
            weight_decay=1e-2,
            nesterov=True,
            maximize=False,
            fused=False,
            foreach=False,
        )

        for _ in range(3):
            opt_fused.zero_grad(set_to_none=True)
            opt_ref.zero_grad(set_to_none=True)

            grad = torch.randn_like(t)
            t.grad = grad
            t_ref.grad = grad.clone()

            opt_fused.step()
            opt_ref.step()

            torch.testing.assert_close(t, t_ref, atol=1e-6, rtol=1e-6)
            torch.musa.synchronize()

    @testing.test_on_nonzero_card_if_multiple_musa_device(1)
    def test_fp32_training(self):
        lr = 1e-2

        model = Model().to("musa")
        model_ref = Model().to("musa")
        model.load_state_dict(copy.deepcopy(model_ref.state_dict()))

        params = [p for p in model.parameters() if p.requires_grad]
        params_ref = [p for p in model_ref.parameters() if p.requires_grad]

        opt_fused = _make_sgd(
            params,
            lr=lr,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            maximize=False,
            fused=True,
            foreach=False,
        )
        opt_ref = _make_sgd(
            params_ref,
            lr=lr,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            maximize=False,
            fused=False,
            foreach=False,
        )

        for _ in range(10):
            x = torch.rand([32, 1, 28, 28], device="musa")
            gt = torch.rand([32, 10], device="musa")

            opt_fused.zero_grad(set_to_none=True)
            opt_ref.zero_grad(set_to_none=True)

            out = model(x)
            loss = ((gt - out) ** 2).mean()
            loss.backward()
            opt_fused.step()

            out_ref = model_ref(x)
            loss_ref = ((gt - out_ref) ** 2).mean()
            loss_ref.backward()
            opt_ref.step()

            # compare weights & grads on key modules
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

            torch.musa.synchronize()

    @pytest.mark.skipif(
        testing.get_musa_arch() < 22, reason="Skipped due to limited support"
    )
    @testing.test_on_nonzero_card_if_multiple_musa_device(1)
    def test_amp_training(self):
        # This mainly exercises grad_scale/found_inf integration paths when present.
        lr = 1e-2

        model = Model().to("musa")
        model_ref = Model().to("musa")
        model.load_state_dict(copy.deepcopy(model_ref.state_dict()))

        params = [p for p in model.parameters() if p.requires_grad]
        params_ref = [p for p in model_ref.parameters() if p.requires_grad]

        opt_fused = _make_sgd(
            params,
            lr=lr,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            maximize=False,
            fused=True,
            foreach=False,
        )
        opt_ref = _make_sgd(
            params_ref,
            lr=lr,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            maximize=False,
            fused=False,
            foreach=False,
        )

        scaler = torch.musa.amp.GradScaler(enabled=True)
        scaler_ref = torch.musa.amp.GradScaler(enabled=True)

        for _ in range(10):
            x = torch.rand([32, 1, 28, 28], device="musa")
            gt = torch.rand([32, 10], device="musa")

            opt_fused.zero_grad(set_to_none=True)
            opt_ref.zero_grad(set_to_none=True)

            with torch.musa.amp.autocast(enabled=True):
                out = model(x)
                loss = ((gt - out) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(opt_fused)
            scaler.update()

            with torch.musa.amp.autocast(enabled=True):
                out_ref = model_ref(x)
                loss_ref = ((gt - out_ref) ** 2).mean()
            scaler_ref.scale(loss_ref).backward()
            scaler_ref.step(opt_ref)
            scaler_ref.update()

            for m, m_ref in zip(model.modules(), model_ref.modules()):
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    torch.testing.assert_close(
                        m.weight, m_ref.weight, atol=1e-3, rtol=1e-3, equal_nan=True
                    )

            torch.musa.synchronize()

    @testing.test_on_nonzero_card_if_multiple_musa_device(1)
    def test_state_dict(self):
        model = Model().to("musa")
        params = [p for p in model.parameters() if p.requires_grad]

        optimizer = _make_sgd(
            params,
            lr=1e-2,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
            maximize=False,
            fused=True,
            foreach=False,
        )

        total_steps = 3
        for _ in range(total_steps):
            x = torch.rand([8, 1, 28, 28], device="musa")
            gt = torch.rand([8, 10], device="musa")
            out = model(x)
            loss = ((gt - out) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, "sgd_state_dict.pth")
            torch.save({"optimizer_state_dict": optimizer.state_dict()}, filename)
            opt_sd = torch.load(filename)["optimizer_state_dict"]

            # SGD w/ momentum should have momentum_buffer for each param
            for st in opt_sd["state"].values():
                assert "momentum_buffer" in st
                # momentum_buffer should live on the same device as params (musa)
                assert st["momentum_buffer"].device.type in ("musa", "privateuseone")
