"""Test Fused LAMB optimzer"""

# pylint: disable=missing-class-docstring, missing-function-docstring
from itertools import product
import torch
from torch.optim import Optimizer
from torch_musa.optim import FusedLAMB
from torch_musa.multi_tensor_apply import multi_tensor_applier
from torch_musa.utils import ext_loader
import torch_musa


class RefLAMB(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        if multi_tensor_applier.available:
            ext_module = ext_loader.load_ext(
                "_ext", ["multi_tensor_l2norm", "multi_tensor_lamb"]
            )
            self.multi_tensor_l2norm = ext_module.multi_tensor_l2norm
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor(
                [0], dtype=torch.int, device=self.param_groups[0]["params"][0].device
            )
            self.multi_tensor_lamb = ext_module.multi_tensor_lamb
        else:
            raise RuntimeError("torch_musa.optim.FusedLAMB requires musa extensions")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # create separate grad lists for fp32 and fp16 params
        g_all_32, g_all_16 = [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dtype == torch.float32:
                    g_all_32.append(p.grad.data)
                elif p.dtype == torch.float16:
                    g_all_16.append(p.grad.data)
                else:
                    raise RuntimeError("FusedLAMB only support fp16 and fp32.")

        device = self.param_groups[0]["params"][0].device
        g_norm_32, g_norm_16 = torch.zeros(1, device=device), torch.zeros(
            1, device=device
        )
        # compute grad norm for two lists
        if len(g_all_32) > 0:
            g_norm_32 = multi_tensor_applier(
                self.multi_tensor_l2norm, self._dummy_overflow_buf, [g_all_32], False
            )[0]
        if len(g_all_16) > 0:
            g_norm_16 = multi_tensor_applier(
                self.multi_tensor_l2norm, self._dummy_overflow_buf, [g_all_16], False
            )[0]
        # blend two grad norms to get global grad norm
        global_grad_norm = multi_tensor_applier(
            self.multi_tensor_l2norm,
            self._dummy_overflow_buf,
            [[g_norm_32, g_norm_16]],
            False,
        )[0]
        max_grad_norm = 1.0
        clipped_ratio = max_grad_norm / max(global_grad_norm, max_grad_norm)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.data *= clipped_ratio
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "FusedLAMB does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["m"] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values
                    state["v"] = torch.zeros_like(p.data)

                m_t, v_t = state["m"], state["v"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                m_t.mul_(beta1).add_(grad, alpha=1 - beta1)
                v_t.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Debiasing
                m_t_hat = m_t / (1.0 - beta1 ** state["step"])
                v_t_hat = v_t / (1.0 - beta2 ** state["step"])

                update = m_t_hat / v_t_hat.sqrt().add(group["eps"])

                if group["weight_decay"] != 0:
                    update.add_(p.data, alpha=group["weight_decay"])

                trust_ratio = 1.0
                w_norm = p.data.pow(2).sum().sqrt()
                g_norm = update.pow(2).sum().sqrt()
                if w_norm > 0 and g_norm > 0:
                    trust_ratio = w_norm / g_norm

                state["w_norm"] = w_norm
                state["g_norm"] = g_norm
                state["trust_ratio"] = trust_ratio

                step_size = group["lr"]

                p.data.add_(update, alpha=-step_size * trust_ratio)

        return loss


class LambTest:
    def __init__(self, max_abs_diff=1e-3, max_rel_diff=1, iters=7):
        self.max_abs_diff = max_abs_diff
        self.max_rel_diff = max_rel_diff
        self.iters = iters
        torch.musa.manual_seed(9876)

    def gen_param_optim(self, tensors, lamb_option):
        ref_param = [torch.nn.Parameter(t.clone()) for t in tensors]
        tst_param = [torch.nn.Parameter(t.clone()) for t in tensors]

        ref_optim = self.ref_optim(ref_param, **lamb_option)
        tst_optim = self.tst_optim(tst_param, use_nvlamb=True, **lamb_option)

        return ref_param, tst_param, ref_optim, tst_optim

    def gen_grad(self, ref_param, tst_param):
        for p_ref, p_tst in zip(ref_param, tst_param):
            p_ref.grad = torch.rand_like(p_ref)
            p_tst.grad = p_ref.grad

    def get_max_diff(self, ref_param, tst_param):
        max_abs_diff = max_rel_diff = 0
        for p_ref, p_tst in zip(ref_param, tst_param):
            max_abs_diff_p = (p_ref - p_tst).abs().max().item()
            # abs(x / y) == abs(x) / abs(y)
            # and if we don't do clamp, the subnormal value would
            # cause a huge value error
            max_rel_diff_p = (
                ((p_ref - p_tst).abs().clamp(min=1e-6) / p_ref.abs().clamp(min=1e-6))
                .max()
                .item()
            )

            max_abs_diff = max(max_abs_diff, max_abs_diff_p)
            max_rel_diff = max(max_rel_diff, max_rel_diff_p)

        return max_abs_diff, max_rel_diff

    def gen_single_type_test(self, param_type=torch.float32, device="musa"):
        nelem = 278011
        tensor = torch.rand(nelem, dtype=param_type, device=device)
        weight_decay = [0, 0.01]

        for wd in weight_decay:
            lamb_option = {
                "lr": 5e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": wd,
            }
            ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(
                [tensor], lamb_option
            )

            for _ in range(self.iters):
                self.gen_grad(ref_param, tst_param)
                ref_optim.step()
                torch.musa.synchronize()
                tst_optim.step()
                torch.musa.synchronize()
                max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)

                assert max_abs_diff <= self.max_abs_diff
                assert max_rel_diff <= self.max_rel_diff


class FusedLAMBTest(LambTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_optim = RefLAMB
        self.tst_optim = FusedLAMB

    def test_float32(self):
        self.gen_single_type_test(param_type=torch.float32)

    def test_multi_device(self):
        if torch_musa.device_count() < 2:
            return
        devices = ("musa:0", "musa:1")
        for current_dev, tensor_dev in product(devices, devices):
            with torch.musa.device(current_dev):
                self.gen_single_type_test(param_type=torch.float32, device=tensor_dev)

    def test_multi_params(self):
        sizes = [[4096, 1024], [4096], [4096, 2048], [32320, 1024], [1]]
        weight_decay = [0, 0.01]

        for wd in weight_decay:
            lamb_option = {
                "lr": 5e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": wd,
            }
            tensors = []
            for size in sizes:
                tensors.append(torch.rand(size, dtype=torch.float, device="musa"))
            ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(
                tensors, lamb_option
            )

            for _ in range(self.iters):
                self.gen_grad(ref_param, tst_param)
                ref_optim.step()
                tst_optim.step()
                max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)

                assert max_abs_diff <= self.max_abs_diff
                assert max_rel_diff <= self.max_rel_diff

    def test_lamb_option(self):
        nelem = 1
        tensor = torch.rand(nelem, dtype=torch.float32, device="musa")
        weight_decay = [0, 0.01]

        for wd in weight_decay:
            lamb_option = {
                "lr": 0.01,
                "betas": (0.6, 0.9),
                "eps": 3e-6,
                "weight_decay": wd,
            }
            ref_param, tst_param, ref_optim, tst_optim = self.gen_param_optim(
                [tensor], lamb_option
            )

            for _ in range(self.iters):
                self.gen_grad(ref_param, tst_param)
                ref_optim.step()
                tst_optim.step()
                max_abs_diff, max_rel_diff = self.get_max_diff(ref_param, tst_param)

                assert max_abs_diff <= self.max_abs_diff
                assert max_rel_diff <= self.max_rel_diff

    def run(self):
        test_funcs = [
            m for m in dir(self) if callable(getattr(self, m)) and m.startswith("test")
        ]
        for func in test_funcs:
            getattr(self, func)()


def test_fused_lamb():
    FusedLAMBTest(max_abs_diff=1e-6).run()
