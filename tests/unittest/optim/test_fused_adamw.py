"""Test FusedAdamW Optimizer"""

# pylint: disable=unused-import,invalid-name,not-callable
import os
import copy
import tempfile
import pytest
import torch

import torch_musa
from torch_musa.optim import FusedAdamW
from torch_musa import testing


# Borrowed from apex test_adam
class Model(torch.nn.Module):
    """Model for testing FusedAdamW"""

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
        """perform model's forward"""
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
        y = self.relu5(y)
        return y


class TestFusedAdamW:
    """class of FusedAdamW unit test suits"""

    @pytest.mark.parametrize("weight_decay", [0.0, 1.0])
    @pytest.mark.parametrize("amsgrad", [False, True])
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
        ],
    )
    def test_basic(self, weight_decay, amsgrad, dtype):
        """test basic functionality of FusedAdamW"""
        t = torch.zeros(1024 * 1024 * 8, dtype=dtype, device="musa")
        t_ref = torch.zeros(1024 * 1024 * 8, dtype=dtype, device="cpu")

        lr = 0.002
        optimizer_fused = FusedAdamW(
            [t], lr=lr, weight_decay=weight_decay, amsgrad=amsgrad
        )
        optimizer_ref = torch.optim.AdamW(
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

    def test_fp32_traing(self):
        """test FusedAdamW in fp32 training mode"""
        lr = 0.001
        model = Model().to("musa")
        model_ref = Model().to("musa")

        model.load_state_dict(copy.deepcopy(model_ref.state_dict()))

        params = [p for p in model.parameters() if p.requires_grad]
        params_ref = [p for p in model_ref.parameters() if p.requires_grad]

        optimizer = FusedAdamW(params, lr=lr)
        optimizer_ref = torch.optim.AdamW(params_ref, lr=lr)

        for _ in range(20):
            x = torch.rand([32, 1, 28, 28], device="musa")
            x_ref = x.clone()
            gt = torch.rand([32, 10], device="musa")
            gt_ref = gt.clone()

            out = model(x)
            loss = ((gt - out) ** 2).mean()
            loss.backward()
            optimizer.step()

            out_ref = model_ref(x_ref)
            loss_ref = ((gt_ref - out_ref) ** 2).mean()
            loss_ref.backward()
            optimizer_ref.step()

            for module in zip(model.modules(), model_ref.modules()):
                m = module[0]
                m_ref = module[1]
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

            # Deliberately set different steps for different parameters
            for i, state in enumerate(optimizer.state.values()):
                state["step"] += i

            for i, state in enumerate(optimizer_ref.state.values()):
                state["step"] += i

            # reset for next iteration
            optimizer.zero_grad()
            optimizer_ref.zero_grad()
            model.load_state_dict(copy.deepcopy(model_ref.state_dict()))

    @pytest.mark.skipif(
        testing.get_musa_arch() < 22, reason="Skipped due to limited support"
    )
    def test_amp_traing(self):
        """Test FusedAdamW in mixed precision training mode"""
        lr = 1e-4
        model = Model().to("musa")
        model_ref = Model().to("musa")
        model.load_state_dict(copy.deepcopy(model_ref.state_dict()))

        scaler = torch.musa.amp.GradScaler(enabled=True)
        scaler_ref = torch.musa.amp.GradScaler(enabled=True)

        params = [p for p in model.parameters() if p.requires_grad]
        params_ref = [p for p in model_ref.parameters() if p.requires_grad]

        optimizer = FusedAdamW(params, lr=lr)
        optimizer_ref = torch.optim.AdamW(params_ref, lr=lr)

        for _ in range(20):
            x = torch.rand([32, 1, 28, 28], device="musa")
            x_ref = x.clone()
            gt = torch.rand([32, 10], device="musa")
            gt_ref = gt.clone()

            with torch.musa.amp.autocast(enabled=True):
                out = model(x)
                loss = ((gt - out) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.musa.amp.autocast(enabled=True):
                out_ref = model_ref(x_ref)
                loss_ref = ((gt_ref - out_ref) ** 2).mean()
            scaler_ref.scale(loss_ref).backward()
            scaler_ref.step(optimizer_ref)
            scaler_ref.update()

            for module in zip(model.modules(), model_ref.modules()):
                m = module[0]
                m_ref = module[1]
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

            # reset for next iteration
            optimizer.zero_grad()
            optimizer_ref.zero_grad()
            model.load_state_dict(copy.deepcopy(model_ref.state_dict()))

    def test_state_dict(self):
        """test load/store FusedAdam's state_dict"""
        model = Model().to("musa")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = FusedAdamW(params, lr=0.001)

        total_steps = 5
        for _ in range(total_steps):
            x = torch.rand([32, 1, 28, 28], device="musa")
            gt = torch.rand([32, 10], device="musa")
            out = model(x)
            loss = ((gt - out) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, "state_dict.pth")
            torch.save({"optimizer_state_dict": optimizer.state_dict()}, filename)
            optimizer_state_dict = torch.load(filename)["optimizer_state_dict"]

            for param_state in optimizer_state_dict["state"].values():
                # host `step` on CPU
                assert param_state["step"].device.type == "cpu"
                assert param_state["step"].item() == total_steps
