"""implement heuristics for triton_musa"""

__all__ = ["_apply_triton_heuristics_patches"]

# pylint:disable=import-outside-toplevel,invalid-name
# pylint:disable=W0613,W0122,W0622
from typing import List, Optional
import copy

import torch
import torch._inductor.runtime
from torch._inductor.runtime.runtime_utils import get_first_attr
from torch._inductor.runtime.triton_heuristics import (
    log,
    unique_configs,
    hash_configs,
    AutotuneCache,
    CachingAutotuner,
    DebugAutotuner,
)
import torch._inductor.runtime.triton_heuristics


# check triton
try:
    import triton
except ImportError:
    triton = None

if triton:
    from triton import Config
    from triton.compiler import CompiledKernel

    try:
        from triton.compiler.compiler import ASTSource
    except ImportError:
        ASTSource = None

    try:
        from triton.backends.compiler import GPUTarget
    except ImportError:
        GPUTarget = None
else:
    Config = object
    ASTSource = None
    GPUTarget = None


# Implement a MUSACachingAutotuner to facilitate performance tuning on MTGPU
class MUSACachingAutotuner(CachingAutotuner):
    """class of MUSACachingAutotuner"""

    def _precompile_config(self, cfg: Config, warm_cache_only: bool):
        """Ahead of time compile a given autotuner config."""

        # cuda and hip configs are removed
        compile_meta = copy.deepcopy(self.triton_meta)
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = self.inductor_meta.get("assert_indirect_indexing", True)

        # device type will be "musa" rather than "cuda" here
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        if ASTSource:
            compile_args = (
                ASTSource(
                    self.fn,
                    compile_meta["signature"],
                    compile_meta["constants"],
                    compile_meta["configs"][0],
                ),
            )

            if compile_meta["cc"] < 31:
                warp_size = 128
            else:
                warp_size = 32

            if GPUTarget:
                target = GPUTarget(
                    compile_meta["device_type"], compile_meta["cc"], warp_size
                )
            else:
                target = (compile_meta["device_type"], compile_meta["cc"])

            options = {
                "num_warps": compile_meta["num_warps"],
                "num_stages": compile_meta["num_stages"],
                "debug": compile_meta["debug"],
            }
            compile_kwargs = {
                "target": target,
                "options": options,
            }
        else:
            compile_args = (self.fn,)
            compile_kwargs = compile_meta

        if warm_cache_only:
            return (
                triton.compile(*compile_args, **compile_kwargs),
                None,
            )

        # importing from torch is safe now that precompile has returned
        from torch._dynamo.device_interface import DeviceGuard

        device_interface = self.get_device_interface()

        # load binary to the correct device
        with DeviceGuard(device_interface, compile_meta["device"]):  # type: ignore[attr-defined]
            # need to initialize context
            device_interface.synchronize(device_interface.current_device())

            try:
                binary = triton.compile(*compile_args, **compile_kwargs)
            except Exception:
                log.exception(
                    "Triton compilation failed: %s\n%s\nmetadata: %s",
                    self.inductor_meta.get("kernel_name", "triton_"),
                    self.fn.src,
                    compile_meta,
                )
                raise
            binary._init_handles()

        call_args = [
            arg
            for i, arg in enumerate(self.fn.arg_names)
            if i not in self.fn.constexprs
        ]
        def_args = [name for name in self.fn.arg_names if name not in cfg.kwargs]

        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "launch_enter_hook": CompiledKernel.launch_enter_hook,
            "launch_exit_hook": CompiledKernel.launch_exit_hook,
            "metadata": (
                binary.packed_metadata
                if hasattr(binary, "packed_metadata")
                else binary.metadata
            ),
            "shared": binary_shared,
        }

        scope["num_warps"] = (
            binary.num_warps
            if hasattr(binary, "num_warps")
            else binary.metadata.num_warps
        )

        scope["cta_args"] = (
            (binary.num_ctas, *get_first_attr(binary, "cluster_dims", "clusterDims"))
            if hasattr(binary, "num_ctas")
            else (
                (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                if hasattr(binary, "metadata")
                else ()
            )
        )

        scope["function"] = get_first_attr(binary, "function", "cu_function")

        def get_launch_args_without_kernel_launch_metadata(
            grid,
            grid_0,
            grid_1,
            grid_2,
            stream,
            function,
            metadata,
            bin,
            launch_enter_hook,
            launch_exit_hook,
            num_warps,
            shared,
            cta_args,
            args,
        ):
            """
            Construct launch args before CompiledKernel.launch_metadata is added.
            """
            return (
                grid_0,
                grid_1,
                grid_2,
                num_warps,
                *cta_args,
                shared,
                stream,
                function,
                launch_enter_hook,
                launch_exit_hook,
                metadata,
            )

        # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
        # `bin.launch_metadata` is relatively expensive, and returns None unless a
        # `launch_enter_hook` is installed.  So if we don't have that hook installed,
        # we want to burn None in to the launch args with zero overhead.
        # See https://github.com/pytorch/pytorch/issues/123597
        if binary.launch_enter_hook:

            def get_launch_args_with_kernel_launch_metadata(
                grid,
                grid_0,
                grid_1,
                grid_2,
                stream,
                function,
                metadata,
                bin,
                launch_enter_hook,
                launch_exit_hook,
                num_warps,
                shared,
                cta_args,
                args,
            ):
                """
                Construct launch args after CompiledKernel.launch_metadata is added
                by https://github.com/openai/triton/pull/3492 .
                """
                return (
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    bin.launch_metadata(grid, stream, *args),
                    launch_enter_hook,
                    launch_exit_hook,
                )

        else:

            def get_launch_args_with_kernel_launch_metadata(
                grid,
                grid_0,
                grid_1,
                grid_2,
                stream,
                function,
                metadata,
                bin,
                launch_enter_hook,
                launch_exit_hook,
                num_warps,
                shared,
                cta_args,
                args,
            ):
                """
                Construct launch args after CompiledKernel.launch_metadata is added
                by https://github.com/openai/triton/pull/3492 .
                """
                return (
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    None,
                    launch_enter_hook,
                    launch_exit_hook,
                )

        scope["get_launch_args"] = (
            get_launch_args_with_kernel_launch_metadata
            if hasattr(binary, "launch_metadata")
            else get_launch_args_without_kernel_launch_metadata
        )

        scope["runner"] = get_first_attr(binary, "run", "c_wrapper")

        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid

                args = {', '.join(call_args)},
                launch_args = get_launch_args(
                    grid, grid_0, grid_1, grid_2, stream, function,
                    metadata, bin, launch_enter_hook, launch_exit_hook,
                    num_warps, shared, cta_args, args
                )
                runner(*launch_args, *args)
                return bin
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = self.fn
            launcher.bin = binary

        return binary, launcher

    def save_gpu_kernel(self, grid, stream, launcher):
        if callable(grid):
            grid_x, grid_y, grid_z = grid(launcher.config.kwargs)
        else:
            grid_x, grid_y, grid_z = grid

        key = self.inductor_meta.get("kernel_name", None)  # unique kernel name
        assert key is not None, "kernel_name can not be None"
        params = {
            "mangled_name": (
                launcher.bin.metadata.name
                if hasattr(launcher.bin.metadata, "name")
                else launcher.bin.metadata["name"]
            ),
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "x_block": launcher.config.kwargs.get("XBLOCK", 1),
            "y_block": launcher.config.kwargs.get("YBLOCK", None),
            "z_block": launcher.config.kwargs.get("ZBLOCK", None),
            "num_warps": (
                launcher.bin.num_warps
                if hasattr(launcher.bin, "num_warps")
                else launcher.bin.metadata.num_warps
            ),
            "shared_mem": (
                launcher.bin.shared
                if hasattr(launcher.bin, "shared")
                else launcher.bin.metadata.shared
            ),
            "stream": stream,
            # User defined triton kernels will have arbitrary kwarg names
            "meta": launcher.config.kwargs,
        }
        from ..codegen.codecache import MusaKernelParamCache

        bin_type = {"musa": "mubin"}.get(self.device_props.type, "cubin")
        binary = launcher.bin.asm[bin_type]
        MusaKernelParamCache.set(key, params, binary)

        self.cuda_kernel_saved = True


class MUSADebugAutotuner(DebugAutotuner, MUSACachingAutotuner):
    pass


def cached_autotune(
    size_hints: Optional[List[int]],
    configs: List[Config],
    triton_meta,
    heuristic_type,
    filename=None,
    inductor_meta=None,
    custom_kernel=False,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename
    inductor_meta = {} if inductor_meta is None else inductor_meta

    disabled = inductor_meta.get("force_disable_caches", False)

    # on disk caching logic and/or remote caching
    autotune_cache = None
    if (
        not disabled
        and filename is not None
        and (len(configs) > 1 or inductor_meta.get("coordinate_descent_tuning"))
    ):
        configs_hash = hash_configs(configs)

        autotune_cache = AutotuneCache.create(inductor_meta, filename, configs_hash)
        if autotune_cache:
            if best_config := autotune_cache.read_best(inductor_meta, configs):
                configs = [best_config]

    else:
        if disabled:
            log.debug("autotune caching is disabled by config.force_disable_caches")

    mutated_arg_names = inductor_meta.pop("mutated_arg_names", ())

    def decorator(fn):
        # Remove XBLOCK from config if it's not a function argument.
        # This way, coordinate descent tuning will not try to tune it.
        #
        # Context: When TritonKernel.no_x_dim is True, we hardcode XBLOCK to 1.
        import inspect

        if "XBLOCK" not in inspect.signature(fn.fn).parameters:
            for tconfig in configs:
                if "XBLOCK" in tconfig.kwargs:
                    assert tconfig.kwargs["XBLOCK"] == 1
                    tconfig.kwargs.pop("XBLOCK")

        if inductor_meta.get("profile_bandwidth"):
            return MUSADebugAutotuner(
                fn,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                regex_filter=inductor_meta["profile_bandwidth_regex"],
                with_profiler=inductor_meta[
                    "profile_bandwidth_with_do_bench_using_profiling"
                ],
                configs=configs,
                save_cache_hook=autotune_cache and autotune_cache.save,
                mutated_arg_names=mutated_arg_names,
                heuristic_type=heuristic_type,
                size_hints=size_hints,
                custom_kernel=custom_kernel,
                filename=filename,
            )
        return MUSACachingAutotuner(
            fn,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            configs=configs,
            save_cache_hook=autotune_cache and autotune_cache.save,
            mutated_arg_names=mutated_arg_names,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
            custom_kernel=custom_kernel,
            filename=filename,
        )

    return decorator


# Maybe we can patch Inductor IRs such as pointwise IR, reduction IR
# here in the near future


def _apply_triton_heuristics_patches():
    torch._inductor.runtime.triton_heuristics.cached_autotune = cached_autotune
    # torch._inductor.runtime.triton_heuristics.CachingAutotuner = MUSACachingAutotuner
