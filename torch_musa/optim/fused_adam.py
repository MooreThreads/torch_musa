"""Fused Adam Optimizer"""

# pylint: disable=C0301,unused-argument

from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)

from ..utils import ext_loader

# check fused_adam method
ext_module = ext_loader.load_ext("_ext", ["fused_adam"])

__all__ = ["FusedAdam"]


class FusedAdam(Optimizer):
    """Fused implementation of Adam algorithm.

    Currently GPU-only.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.


    FusedAdam impements the same functionality as torch.optim.Adam(fused=True),
    the difference is that 'state_steps' is located on the CPU in our implementation.


    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximinze (bool, optional): maximize the params based on the objective,
            instead of minimizing (default: False)
        capturable (bool, optional): whether to use the version of the optimizer
            that can be used with MUSA Graphs. (default: False)
        differentiable (bool, optional): whether autograd should occur through the
            optimizer step in training. Otherwise, the step() function runs in a
            torch.no_grad() context. Setting to True can impair performance, so leave
            it False if you don't intend to run autograd through this instance (default: False)


    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    Example:
        >>> from torch_musa.optim import FusedAdam
        >>> optimizer = FusedAdam(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if differentiable:
            raise RuntimeError("FusedAdam does not support differentiable=True")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize,
            "capturable": capturable,
            "differentiable": differentiable,
            "fused": True,
        }
        super().__init__(params, defaults)

        if differentiable:
            raise RuntimeError("FusedAdam does not support `differentiable`")
        self._step_supports_amp_scaling = True
        fused_supported_devices = [
            torch._C._get_privateuse1_backend_name(),
        ]  # musa only
        if not all(
            p.device.type in fused_supported_devices and torch.is_floating_point(p)
            for pg in self.param_groups
            for p in pg["params"]
        ):
            raise RuntimeError(
                "FusedAdam requires all the params to be floating point Tensors of "
                f"supported devices: {fused_supported_devices}."
            )

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", True)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]), dtype=torch.float32)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):

        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse gradients, please consider SparseAdam instead"
                    )
                grads.append(p.grad)

                state = self.state[p]

                # lazy state initialization
                if len(state) == 0:
                    # NOTE: same step within same group now
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError(
                        "`requires_grad` is not supported for `step` in differentiable mode"
                    )

                state_steps.append(state["step"])

        return has_complex

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:  # list
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            func = (
                _fused_adam_tensor_lr
                if isinstance(group["lr"], torch.Tensor)
                else _fused_adam
            )
            func(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                group["lr"],
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def _fused_adam_impl(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    lr: Union[float, Tensor],
    capturable: bool = False,
    differentiable: bool = False,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
) -> None:
    if not params:
        return

    if torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with FusedAdam")

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    grad_scale_dict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else None
    )
    found_inf_dict = {found_inf.device: found_inf} if found_inf is not None else None

    lr_dict = (
        {lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != "cpu" else None
    )

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]
    )

    for (device, _), (
        (
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
        ),
        _,
    ) in grouped_tensors.items():
        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            if device not in grad_scale_dict:
                grad_scale_dict[device] = grad_scale.to(device, non_blocking=True)
            device_grad_scale = grad_scale_dict[device]
        if found_inf is not None:
            if found_inf not in found_inf_dict:
                found_inf_dict[device] = found_inf.to(device, non_blocking=True)
            device_found_inf = found_inf_dict[device]
        if lr_dict is not None and device not in lr_dict:
            lr_dict[device] = lr.to(device=device, non_blocking=True)
            lr = lr_dict[device]  # tensor lr
        ext_module.fused_adam(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            amsgrad,
            maximize,
            device_grad_scale,
            device_found_inf,
        )

        if device_found_inf is not None:
            # running on cpu should be ok
            device_found_inf_ = device_found_inf.cpu().reshape_as(device_state_steps[0])
            torch._foreach_sub_(
                device_state_steps, [device_found_inf_] * len(device_state_steps)
            )


# note(mingyuan.wang): To work with PyTorch Dispatch System, we need to register our fused adam operator,
# This enables automatic handling of tensor subclasses (e.g., DTensor or WeightWithDynamicFloat8CastTensor)
# via the __torch_dispatch__ mechanism.
# Without registration, manual unwrapping of input tensors would be required, which is not convenient for maintenance.
@torch.library.custom_op(
    "musa::_fused_adam_",
    mutates_args=(
        "params",
        "grads",
        "exp_avgs",
        "exp_avg_sqs",
        "max_exp_avg_sqs",
        "state_steps",
    ),
    device_types="musa",
)
def _fused_adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    lr: float,
    capturable: bool = False,
    differentiable: bool = False,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
) -> None:

    _fused_adam_impl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
    )


@torch.library.custom_op(
    "musa::_fused_adam_.tensor_lr",
    mutates_args=(
        "params",
        "grads",
        "exp_avgs",
        "exp_avg_sqs",
        "max_exp_avg_sqs",
        "state_steps",
    ),
    device_types="musa",
)
def _fused_adam_tensor_lr(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    lr: Tensor,
    capturable: bool = False,
    differentiable: bool = False,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
) -> None:

    _fused_adam_impl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
    )
