"""Heuristics for triton"""

# pylint: disable=invalid-name,unused-import
import os
import copy
import json
import functools
import logging
import operator
import inspect
from typing import (
    Optional,
    List,
    Callable,
    Any,
)

import torch
from torch._inductor import config
from torch._inductor.triton_heuristics import (
    log,
    AutotuneHint,
    CachingAutotuner,
    HeuristicType,
    hash_configs,
    load_cached_autotuning,
    unique_configs,
    triton_config,
    triton_config_reduction,
    autotune_hints_to_configs,
    disable_pointwise_autotuning,
)
from torch._inductor.ir import TileHint, ReductionHint

from torch_musa._MUSAC import _musa_getCurrentRawStream as get_musa_stream


from .utils import triton_is_available, do_bench


if triton_is_available():
    import triton
    from triton import Config
    from triton.runtime.autotuner import OutOfResources
else:
    Config = object
    triton = None
    OutOfResources = object


class MUSACachingAutotuner(CachingAutotuner):
    """CachingAutotuner for MUSA

    Rewrite some methods that hard coded with cuda.
    """

    def precompile(self, warm_cache_only_with_cc=None):
        with self.lock:
            if self.launchers:
                return
            self.launchers = []
            compiled_binaries = []
            for c in self.configs:
                try:
                    compiled_binary, launcher = self._precompile_config(
                        c, warm_cache_only_with_cc
                    )
                except OutOfResources:  # pylint: disable=catching-non-exception
                    # Skip the config if we run out of resource
                    continue
                self.launchers.append(launcher)
                compiled_binaries.append(compiled_binary)

            if len(self.launchers) == 0:
                raise RuntimeError(
                    "No valid triton configs. Report a fatal compilation error"
                )

            # TODO(mingyuan.wang): add optimization that avoid register spill (mainly REDUCTION IR)
            self.configs = None

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: Optional[int]):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.triton_meta)
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = config.assert_indirect_indexing
        compile_meta["device_type"] = "musa"

        if warm_cache_only_with_cc:
            return (
                triton.compile(
                    self.fn,
                    warm_cache_only=True,
                    cc=warm_cache_only_with_cc,
                    **compile_meta,
                ),
                None,
            )

        # load binary to the correct device
        with torch.musa.device(compile_meta["device"]):
            # need to initialize context
            torch.musa.synchronize(torch.musa.current_device())
            binary = triton.compile(
                self.fn,
                **compile_meta,
            )
            binary._init_handles()

        call_args = [
            arg
            for i, arg in enumerate(self.fn.arg_names)
            if i not in self.fn.constexprs
        ]
        def_args = [name for name in self.fn.arg_names if name not in cfg.kwargs]

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "torch": torch,
            "set_device": torch.musa.set_device,
            "current_device": torch.musa.current_device,
        }
        # pylint: disable=exec-used
        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid

                if hasattr(bin, "num_ctas"):
                    bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps,
                                bin.num_ctas, *bin.clusterDims, bin.shared,
                                stream, bin.cu_function, None, None, None,
                                {', '.join(call_args)})
                else:
                    bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared,
                                stream, bin.cu_function, None, None, None,
                                {', '.join(call_args)})
                return bin
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = getattr(binary, "shared", None)
        launcher.store_cubin = config.triton.store_cubin
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = self.fn
            launcher.bin = binary

        return binary, launcher

    def bench(self, launcher, *args, grid, **kwargs):
        """Measure the performance of a given launcher"""
        # if launcher.n_spills > config.triton.spill_threshold:
        #     log.debug(
        #         "Skip config %s because of register spilling: %d",
        #         launcher.config,
        #         launcher.n_spills,
        #     )
        #     return float("inf")

        stream = get_musa_stream(torch.musa.current_device())

        def kernel_call():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook(
                    {**dict(zip(self.arg_names, args)), **launcher.config.kwargs}
                )

            cloned_args, cloned_kwargs = self.clone_args(*args, **kwargs)
            launcher(
                *cloned_args,
                **cloned_kwargs,
                grid=grid,
                stream=stream,
            )

        return do_bench(kernel_call, rep=40, fast_flush=True)


def cached_autotune(
    size_hints: Optional[List[int]],
    configs: List[Config],
    triton_meta,
    heuristic_type,
    filename=None,
    inductor_meta=None,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename
    save_cache_hook: Optional[Callable[[Any, Any], Any]]
    inductor_meta = {} if inductor_meta is None else inductor_meta

    # on disk caching logic
    if filename is not None and (len(configs) > 1 or config.coordinate_descent_tuning):
        cache_filename = os.path.splitext(filename)[0] + ".best_config"
        configs_hash = hash_configs(configs)
        best_config = load_cached_autotuning(cache_filename, configs_hash, configs)
        if best_config:
            configs = [best_config]

        def save_cache_hook(cfg, found_by_coordesc=False):
            with open(cache_filename, "w", encoding="utf-8") as fd:
                fd.write(
                    json.dumps(
                        {
                            **cfg.kwargs,
                            "num_warps": cfg.num_warps,
                            "num_stages": cfg.num_stages,
                            "configs_hash": configs_hash,
                            "found_by_coordesc": found_by_coordesc,
                        }
                    )
                )
            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, cache_filename)

    else:
        save_cache_hook = None

    mutated_arg_names = inductor_meta.pop("mutated_arg_names", ())

    def decorator(fn):
        # Remove XBLOCK from config if it's not a function argument.
        # This way, coordinate descent tuning will not try to tune it.
        #
        # Context: When TritonKernel.no_x_dim is True, we hardcode XBLOCK to 1.

        if "XBLOCK" not in inspect.signature(fn.fn).parameters:
            for tconfig in configs:
                if "XBLOCK" in tconfig.kwargs:
                    assert tconfig.kwargs["XBLOCK"] == 1
                    tconfig.kwargs.pop("XBLOCK")

        if config.profile_bandwidth:
            raise NotImplementedError("MUSADebugAutotuner is not supported yet")
        return MUSACachingAutotuner(
            fn,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            configs=configs,
            save_cache_hook=save_cache_hook,
            mutated_arg_names=mutated_arg_names,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
        )

    return decorator


def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta

    numel = functools.reduce(operator.mul, size_hints)
    bs = max(256, min(numel // 128, 1024))

    hinted_configs = autotune_hints_to_configs(
        inductor_meta.get("autotune_hints", set()), size_hints, bs
    )

    triton_config_with_settings = functools.partial(
        triton_config, min_elem_per_thread=min_elem_per_thread
    )

    if len(size_hints) == 1:
        if disable_pointwise_autotuning() and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            return cached_autotune(
                size_hints,
                [triton_config_with_settings(size_hints, bs)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )

        return cached_autotune(
            size_hints,
            [
                triton_config_with_settings(size_hints, bs, num_elements_per_warp=256),
                triton_config_with_settings(
                    size_hints, bs // 2, num_elements_per_warp=64
                ),
                *hinted_configs,
            ],
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            heuristic_type=HeuristicType.POINTWISE,
            filename=filename,
        )
    if len(size_hints) == 2:
        if (disable_pointwise_autotuning() or tile_hint == TileHint.SQUARE) and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            return cached_autotune(
                size_hints,
                [triton_config_with_settings(size_hints, 32, 32)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                triton_config_with_settings(size_hints, 32, 32),
                triton_config_with_settings(size_hints, 64, 64),  # ~8% better for fp16
                triton_config_with_settings(size_hints, 256, 16),
                triton_config_with_settings(size_hints, 16, 256),
                triton_config_with_settings(size_hints, bs, 1),
                triton_config_with_settings(size_hints, 1, bs),
                *hinted_configs,
            ],
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    if len(size_hints) == 3:
        if disable_pointwise_autotuning():
            return cached_autotune(
                size_hints,
                [triton_config_with_settings(size_hints, 16, 16, 16)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                triton_config_with_settings(size_hints, 16, 16, 16),
                triton_config_with_settings(size_hints, 64, 8, 8),
                triton_config_with_settings(size_hints, 8, 64, 8),
                triton_config_with_settings(size_hints, 8, 8, 64),
                triton_config_with_settings(size_hints, bs, 1, 1),
                triton_config_with_settings(size_hints, 1, bs, 1),
                triton_config_with_settings(size_hints, 1, 1, bs),
                *hinted_configs,
            ],
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """args to @triton.heuristics()"""
    inductor_meta = {} if inductor_meta is None else inductor_meta

    assert triton_meta is not None
    rnumel = size_hints[-1]
    if len(size_hints) == 2:
        contiguous_config = triton_config_reduction(
            size_hints, 1, (rnumel if 256 <= rnumel < 2048 else 2048)
        )
        outer_config = triton_config_reduction(size_hints, 64, 8)
        tiny_config = triton_config_reduction(
            size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, min(rnumel, 2048)
        )
        if config.max_autotune or config.max_autotune_pointwise:
            pass  # skip all these cases
        elif reduction_hint == ReductionHint.INNER:
            return cached_autotune(
                size_hints,
                [contiguous_config],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        elif reduction_hint == ReductionHint.OUTER:
            return cached_autotune(
                size_hints,
                [outer_config],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return cached_autotune(
                size_hints,
                [tiny_config],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        if disable_pointwise_autotuning():
            return cached_autotune(
                size_hints,
                [triton_config_reduction(size_hints, 32, 128)],
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                contiguous_config,
                outer_config,
                tiny_config,
                triton_config_reduction(size_hints, 64, 64),
                triton_config_reduction(size_hints, 8, 512),
                # halve the XBLOCK/RBLOCK compared to outer_config
                # TODO: this may only be beneficial when each iteration of the reduction
                # is quite heavy.
                # E.g. https://gist.github.com/shunting314/189a8ef69f90db9d614a823385147a72
                triton_config_reduction(size_hints, 64, 4, num_warps=8),
            ],
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            filename=filename,
            heuristic_type=HeuristicType.REDUCTION,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def persistent_reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """args to @triton.heuristics()"""
    xnumel, rnumel = size_hints

    configs = [
        triton_config_reduction(size_hints, xblock, rnumel)
        for xblock in (1, 8, 32, 128)
        if rnumel * xblock <= 4096 and xblock <= xnumel
    ]

    # TODO(jansel): we should be able to improve these heuristics
    if reduction_hint == ReductionHint.INNER and rnumel >= 256:
        configs = configs[:1]
    elif reduction_hint == ReductionHint.OUTER:
        configs = configs[-1:]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        configs = [
            triton_config_reduction(
                size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, rnumel
            )
        ]
    for c in configs:
        # we don't need RBLOCK for persistent reduction
        c.kwargs.pop("RBLOCK")

    if disable_pointwise_autotuning():
        configs = configs[:1]

    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )


def template(num_stages, num_warps, triton_meta, filename=None, inductor_meta=None):
    """
    Compile a triton template
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=num_stages, num_warps=num_warps)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


def user_autotune(configs, triton_meta, filename=None, inductor_meta=None):
    """
    Compile a user defined triton kernel
    """
    defaults = inspect.signature(triton.Config).parameters
    default_num_stages = defaults["num_stages"].default
    default_num_warps = defaults["num_warps"].default

    if len(configs) == 0:
        configs = [
            triton.Config(
                {}, num_stages=default_num_stages, num_warps=default_num_warps
            )
        ]
    else:
        configs = [
            triton.Config(
                c.get("kwargs", {}),
                num_stages=c.get("num_stages", default_num_stages),
                num_warps=c.get("num_warps", default_num_warps),
            )
            for c in configs
        ]

    return cached_autotune(
        None,
        configs,
        triton_meta=triton_meta,
        heuristic_type=HeuristicType.USER_AUTOTUNE,
        filename=filename,
        inductor_meta=inductor_meta,
    )


def foreach(triton_meta, num_warps, filename=None, inductor_meta=None):
    """
    Compile a triton foreach kernel
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=1, num_warps=num_warps)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )
