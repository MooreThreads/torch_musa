"""Fused AdamW Optimizer"""

# pylint: disable=C0301,unused-argument

from collections import defaultdict
from collections.abc import Hashable, Iterable
from copy import deepcopy
from itertools import chain
from typing import Any, Optional, Union, Tuple, List
from typing_extensions import TypeAlias

import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)

from ..utils import ext_loader

StateDict: TypeAlias = dict[str, Any]

# check fused_adamw method
ext_module = ext_loader.load_ext("_ext", ["fused_adamw"])

__all__ = ["FusedAdamW"]


class FusedAdamW(Optimizer):
    """Fused implementation of AdamW algorithm.

    Currently GPU-only.

    This version of fused AdamW implements 2 fusions.

      * Fusion of the AdamW update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.


    FusedAdamW impements the same functionality as torch.optim.AdamW(fused=True),
    the difference is that 'state_steps' is located on the CPU in our implementation.


    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (default: 1e-2)
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


    .. Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    Example:
        >>> from torch_musa.optim import FusedAdamW
        >>> optimizer = FusedAdamW(model.parameters(), lr=0.1, momentum=0.9)
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
        weight_decay: float = 1e-2,
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
            raise RuntimeError("FusedAdamW does not support differentiable=True")

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
            raise RuntimeError("FusedAdamW does not support `differentiable`")
        self._step_supports_amp_scaling = True
        # parameters are allowed to be placed on the CPU during the initialization,
        # but when running optimizer.step(), we will check again whether they are on
        # the GPU device at this time
        fused_supported_devices = [
            torch._C._get_privateuse1_backend_name(),
            "cpu",
        ]  # musa only
        if not all(
            p.device.type in fused_supported_devices and torch.is_floating_point(p)
            for pg in self.param_groups
            for p in pg["params"]
        ):
            raise RuntimeError(
                "FusedAdamW requires all the params to be floating point Tensors of "
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

    # pylint: disable-all
    @staticmethod
    def _process_value_according_to_param_policy(
        param: torch.Tensor,
        value: torch.Tensor,
        param_id: int,
        param_groups: list[dict[Any, Any]],
        key: Hashable = None,
    ) -> torch.Tensor:
        # Floating-point types are a bit special here. They are the only ones
        # that are assumed to always match the type of params.
        # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
        # UNLESS fused or capturable, see note [special device hosting for step]
        fused = False
        capturable = False
        assert param_groups is not None
        for pg in param_groups:
            if param_id in pg["params"]:
                fused = pg["fused"] if "fused" in pg else False
                capturable = pg["capturable"] if "capturable" in pg else False
                break
        if key == "step":
            if capturable or fused:
                return value.to(dtype=torch.float32)
            else:
                return value
        else:
            if param.is_floating_point():
                return value.to(dtype=param.dtype, device=param.device)
            else:
                return value.to(device=param.device)

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: StateDict) -> None:
        r"""Load the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.

        .. note::
            The names of the parameters (if they exist under the "param_names" key of each param group
            in :meth:`state_dict`) will not affect the loading process.
            To use the parameters' names for custom cases (such as when the parameters in the loaded state dict
            differ from those initialized in the optimizer),
            a custom ``register_load_state_dict_pre_hook`` should be implemented to adapt the loaded dict
            accordingly.
            If ``param_names`` exist in loaded state dict ``param_groups`` they will be saved and override
            the current names, if present, in the optimizer state. If they do not exist in loaded state dict,
            the optimizer ``param_names`` will remain unchanged.
        """
        # shallow copy, to be consistent with module API
        state_dict = state_dict.copy()

        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        # Validate the state_dict
        groups = self.param_groups

        # Deepcopy as we write into saved_groups later to update state
        saved_groups = deepcopy(state_dict["param_groups"])

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )

        def _cast(param, value, param_id=None, param_groups=None, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                return FusedAdamW._process_value_according_to_param_policy(
                    param, value, param_id, param_groups, key
                )
            elif isinstance(value, dict):
                return {
                    k: _cast(
                        param, v, param_id=param_id, param_groups=param_groups, key=k
                    )
                    for k, v in value.items()
                }
            elif isinstance(value, Iterable):
                return type(value)(_cast(param, v, param_id=param_id, param_groups=param_groups) for v in value)  # type: ignore[call-arg]
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state: defaultdict[torch.Tensor, dict[Any, Any]] = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = _cast(
                    param, v, param_id=k, param_groups=state_dict["param_groups"]
                )
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(
            group: dict[str, Any], new_group: dict[str, Any]
        ) -> dict[str, Any]:
            new_group["params"] = group["params"]
            if "param_names" in group and "param_names" not in new_group:
                new_group["param_names"] = group["param_names"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

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
            if p.device.type != "musa":
                raise RuntimeError("all parameters should on MUSA device")
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "FusedAdamW does not support sparse gradients, please consider SparseAdamW instead"
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

    @torch.no_grad()
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
                _fused_adamw_tensor_lr
                if isinstance(group["lr"], torch.Tensor)
                else _fused_adamw
            )
            if len(params_with_grad) == 0:
                continue
            func(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def _fused_adamw_impl(
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
    """implementation of fused adamw"""

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

        ext_module.fused_adamw(
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


# note(mingyuan.wang): To work with PyTorch Dispatch System, we need to register our fused adamw operator,
# This enables automatic handling of tensor subclasses (e.g., DTensor or WeightWithDynamicFloat8CastTensor)
# via the __torch_dispatch__ mechanism.
# Without registration, manual unwrapping of input tensors would be required, which is not convenient for maintenance.
@torch.library.custom_op(
    "musa::_fused_adamw_",
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
def _fused_adamw(
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
    _fused_adamw_impl(
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
    "musa::_fused_adamw_.tensor_lr",
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
def _fused_adamw_tensor_lr(
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
    _fused_adamw_impl(
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
