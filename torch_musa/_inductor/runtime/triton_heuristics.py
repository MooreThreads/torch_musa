# pylint: disable=W0221,E1123

"""implement heuristics for triton_musa"""

__all__ = ["_apply_triton_heuristics_patches"]

# pylint:disable=import-outside-toplevel,invalid-name
# pylint:disable=W0613,W0122,W0622
from typing import List, Optional
import copy

import torch
import torch._inductor.runtime
from torch._inductor.runtime.runtime_utils import (
    triton_hash_to_path_key,
)
from torch._inductor.runtime.triton_heuristics import (
    log,
    unique_configs,
    hash_configs,
    AutotuneCache,
    CachingAutotuner,
    DebugAutotuner,
    TritonCompileResult,
    config_to_dict,
)
import torch._inductor.runtime.triton_heuristics
from torch._inductor.runtime import triton_helpers
from torch._inductor.triton_bundler import TritonBundler

# check triton
try:
    import triton
except ImportError:
    triton = None

if triton:
    from triton import Config

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

    def _precompile_config(self, cfg: Config):
        """Ahead of time compile a given autotuner config."""

        # cuda and hip configs are removed
        compile_meta = copy.deepcopy(self.triton_meta)
        cfg_kwargs = cfg.kwargs
        compile_meta["constants"].update(cfg_kwargs)
        for i in self.fn.constexprs:
            arg_name = self.fn.arg_names[i]
            if arg_name not in compile_meta["constants"] and (
                arg_name in ("num_warps", "num_stages")
            ):
                compile_meta["constants"][arg_name] = getattr(cfg, arg_name)
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = self.inductor_meta.get("assert_indirect_indexing", True)

        # device type will be "musa" rather than "cuda" here
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        triton_helpers.set_driver_to_gpu()
        if not ASTSource:
            raise RuntimeError("Installed triton version too old, please upgrade")

        compile_args = (
            ASTSource(
                self.fn,
                compile_meta["signature"],
                compile_meta["constants"],
                compile_meta["configs"][0],
            ),
        )

        target = GPUTarget(
            compile_meta["device_type"],
            compile_meta["cc"],
            128 if compile_meta["cc"] < 31 else 32,
        )

        options = {
            "num_warps": compile_meta["num_warps"],
            "num_stages": compile_meta["num_stages"],
            "debug": compile_meta["debug"],
            "sanitize_overflow": False,  # turn off additional asserts added for overflow checks
        }

        compile_kwargs = {
            "target": target,
            "options": options,
        }

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

        TritonBundler.put(
            triton_hash_to_path_key(binary.hash), self.triton_meta.get("device", 0)
        )
        return TritonCompileResult(binary, cfg, compile_meta, self.inductor_meta)

    def save_gpu_kernel(self, stream, launcher):
        key = self.inductor_meta.get("kernel_name", None)  # unique kernel name
        assert key is not None, "kernel_name can not be None"
        params = {
            "mangled_name": (
                launcher.bin.metadata.name
                if hasattr(launcher.bin.metadata, "name")
                else launcher.bin.metadata["name"]
            ),
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
            "config": config_to_dict(launcher.config),
            "inductor_meta": self.inductor_meta,
            "triton_meta": self.triton_meta,
            "def_args": launcher.def_args,
            "call_args": launcher.call_args,
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
            optimize_mem=True,
            custom_kernel=custom_kernel,
            filename=filename,
        )

    return decorator


# Maybe we can patch Inductor IRs such as pointwise IR, reduction IR
# here in the near future


def _apply_triton_heuristics_patches():
    torch._inductor.runtime.triton_heuristics.cached_autotune = cached_autotune
    # torch._inductor.runtime.triton_heuristics.CachingAutotuner = MUSACachingAutotuner
