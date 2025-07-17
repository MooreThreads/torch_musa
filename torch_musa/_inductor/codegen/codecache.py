"""AOTI compiler"""

from typing import Dict, Optional, Union, cast
import os
import sys
import re
import logging
import struct
import ctypes
import resource
from filelock import FileLock

import torch
from torch._inductor import config
from torch._inductor.package import package_aoti
from torch._inductor.utils import ALIGN_BYTES, _align
from torch._inductor.cpp_builder import (
    CppBuilder,
    get_name_and_dir_from_output_file_path,
)
from torch._inductor.codecache import (
    split_aot_inductor_output_path,
    code_hash,
    write,
    pick_vec_isa,
    run_command_and_check,
    get_lock_dir,
    LOCK_TIMEOUT,
    compile_file,
)
from torch_musa.utils.musa_extension import _find_musa_home
from ..cpp_builder import CppTorchMusaOptions


output_code_log = torch._logging.getArtifactLogger(__name__, "output_code")
log = logging.getLogger(__name__)


def get_cpp_wrapper_mubin_path_name():
    return "mubin_path"  # TODO: ignore this func


def get_hash(content: Union[str, bytes], extra: str = "", hash_type: str = "code"):
    if hash_type == "code":
        return code_hash(content, extra)
    if hash_type in ["cubin", "hsaco", "mubin"]:
        return code_hash(repr(content))
    raise AssertionError(f"Unknown hash type {hash_type}")


class MusaKernelParamCache:
    """
    MUSA KernelParamCache Class
    """

    cache: Dict[str, Dict[str, str]] = {}
    clear = staticmethod(cache.clear)

    @classmethod
    def set(cls, key: str, params: Dict[str, str], cubin: str) -> None:
        """
        key: cache key.
        params: cache params.
        cubin:  mubin path.
        """
        bin_type = "mubin"
        _, path = write(
            cubin,
            bin_type,
            hash_type=bin_type,
            specified_dir=split_aot_inductor_output_path(
                config.aot_inductor.output_path
            )[0],
        )

        params[get_cpp_wrapper_mubin_path_name()] = path

        cls.cache[key] = params

    @classmethod
    def get(cls, key: str) -> Optional[Dict[str, str]]:
        return cls.cache.get(key, None)


class AotCodeCompiler:
    """
    MUSA C++ compile class, use to compile generated MUSA C++ Code to .so
    """

    @classmethod
    def compile(
        cls,
        graph,  # ignore type_check because of cycle link
        source_code: str,
        serialized_extern_kernel_nodes: Optional[str],
        musa: bool,
    ) -> str:
        """
        compile generated MUSA C++ code to .so.

        graph: GraphLowering class
        source_code: generated MUSA C++ code.
        serialized_extern_kernel_nodes: triton kernel binary file.
        musa: if is musa device
        """

        _find_musa_home()  # cpp_extension consults the env

        picked_vec_isa = pick_vec_isa()
        vec_isa_cmd_gen = CppBuilder(
            name="o",
            sources="i",
            BuildOption=CppTorchMusaOptions(
                vec_isa=picked_vec_isa,
                musa=musa,
                aot_mode=graph.aot_mode,
            ),
        )
        cpp_command = repr(vec_isa_cmd_gen.get_command_line())

        fbcode_aot_cpu_re = False
        use_absolute_path = False

        ld_command = "ld"
        objcopy_command = "objcopy"

        (
            specified_output_path,
            specified_so_name,
        ) = split_aot_inductor_output_path(config.aot_inductor.output_path)
        key, input_path = write(
            source_code,
            "cpp",
            extra=cpp_command,
            specified_dir=specified_output_path,
        )
        output_code_log.info("Output code written to: %s", input_path)
        torch._logging.trace_structured(
            "graph_dump",
            lambda: {
                "name": "inductor_aot_code",
                "type": "cpp",
                "filename": input_path,
            },
            payload_fn=lambda: source_code,
        )

        consts_specified_dir = os.path.join(os.path.split(input_path)[0], key)

        def _compile_consts_linux(consts: bytes) -> str:
            _, consts_path = write(
                consts,
                "bin",
                specified_dir=consts_specified_dir,
            )

            consts_o = os.path.splitext(consts_path)[0] + ".o"

            cmd = f"{ld_command} -r -b binary -o {consts_o} {consts_path}"
            run_command_and_check(cmd)
            log.debug("aot constant binary command: %s", cmd)

            if graph.mutated_buffers & set(graph.constants.keys()):
                if len(consts) > 2_000_000_000:
                    raise ValueError(
                        (
                            "Models with buffer mutation included "
                            "doesn't support constants greater than 2GB!"
                        )
                    )
                rename_data = " .data=.ldata"
            else:
                rename_data = " .data=.lrodata,alloc,load,readonly,data,contents"

            assert (
                ALIGN_BYTES & (ALIGN_BYTES - 1)
            ) == 0 and ALIGN_BYTES >= 64, "must be power of 2 and >= 64"
            cmd = (
                f"{objcopy_command} --rename-section"
                f"{rename_data}"
                f" --set-section-alignment .data={ALIGN_BYTES}"
                f" {consts_o} {consts_o}"
            )
            log.debug("aot constant rename section command: %s", cmd)
            run_command_and_check(cmd)

            cmd = f"rm {consts_path}"
            log.debug("aot constant bin removal command: %s", cmd)
            run_command_and_check(cmd)

            body = re.sub(r"[\W]", "_", consts_path)

            symbol_list = []
            symbol_list.append(
                (
                    f"{objcopy_command} --redefine-sym _binary_{body}_start="
                    f"_binary_constants_bin_start {consts_o}"
                )
            )
            symbol_list.append(
                f"{objcopy_command} --redefine-sym _binary_{body}_size="
                f"_binary_constants_bin_size {consts_o}"
            )
            symbol_list.append(
                f"{objcopy_command} --redefine-sym _binary_{body}_end="
                f"_binary_constants_bin_end {consts_o}"
            )
            log.debug("aot constant binary redefine symbol: %s", " ".join(symbol_list))
            for cmd in symbol_list:
                run_command_and_check(cmd)
            return consts_o

        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            if serialized_extern_kernel_nodes:
                output_json = os.path.splitext(input_path)[0] + ".json"
                with open(output_json, "w", encoding="utf-8") as f:
                    f.write(serialized_extern_kernel_nodes)

            output_so = (
                config.aot_inductor.output_path
                if specified_so_name
                else os.path.splitext(input_path)[0] + ".so"
            )

            output_o = os.path.splitext(input_path)[0] + ".o"

            all_cuda = all(
                graph.get_original_value_of_constant(name).is_cuda
                for name in graph.constants.keys()
                if name not in graph.folded_constants
            )

            def get_nbytes_of_tensor(tensor: torch.Tensor, all_cuda: bool) -> int:
                n_bytes = (
                    torch.ops.mkldnn._nbytes(tensor)
                    if tensor.is_mkldnn
                    else tensor.untyped_storage().nbytes()
                )
                return n_bytes if all_cuda else _align(n_bytes)

            consts_size = sum(
                get_nbytes_of_tensor(tensor, all_cuda)
                for (name, tensor) in graph.constants.items()
                if name not in graph.folded_constants
            )
            # TODO: Fix mmap weights with cuda
            use_mmap_weights = not config.is_fbcode() and consts_size > 2_000_000_000
            if config.aot_inductor.force_mmap_weights:
                use_mmap_weights = True

            if config.aot_inductor.package:
                (
                    object_output_name,
                    object_output_dir,
                ) = get_name_and_dir_from_output_file_path(input_path)
                object_build_options = CppTorchMusaOptions(
                    vec_isa=picked_vec_isa,
                    musa=musa,
                    aot_mode=graph.aot_mode,
                    compile_only=True,
                    use_absolute_path=use_absolute_path,
                    use_mmap_weights=use_mmap_weights,
                )
                object_builder = CppBuilder(
                    name=object_output_name,
                    sources=input_path,
                    output_dir=object_output_dir,
                    BuildOption=object_build_options,
                )
                compile_cmd = object_builder.get_command_line()
                output_o = object_builder.get_target_file_path()

                compile_flags = os.path.splitext(input_path)[0] + "_compile_flags.json"
                object_build_options.save_flags_to_file(compile_flags)

            else:
                (
                    object_output_name,
                    object_output_dir,
                ) = get_name_and_dir_from_output_file_path(input_path)
                object_build_options = CppTorchMusaOptions(
                    vec_isa=picked_vec_isa,
                    musa=musa,
                    aot_mode=graph.aot_mode,
                    compile_only=True,
                    use_absolute_path=use_absolute_path,
                    use_mmap_weights=use_mmap_weights,
                )
                object_builder = CppBuilder(
                    name=object_output_name,
                    sources=input_path,
                    output_dir=object_output_dir,
                    BuildOption=object_build_options,
                )
                compile_cmd = object_builder.get_command_line()
                output_o = object_builder.get_target_file_path()

                log.debug("aot compilation command: %s", compile_cmd)

                run_command_and_check(compile_cmd)

            def _to_bytes(t: torch.Tensor, all_cuda: bool) -> bytes:
                def _pad_to_alignment(raw_bytes: bytes) -> bytes:
                    padded_bytes = raw_bytes.ljust(
                        (len(raw_bytes) + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES,
                        b"\x00",
                    )
                    return padded_bytes

                # This serializes the tensor's untyped_storage to bytes by accessing
                # the raw data of the underlying structure.

                if t.numel() == 0:
                    return b""

                if t.is_mkldnn:
                    data_ptr = torch.ops.mkldnn.data_ptr(t)
                    nbytes = torch.ops.mkldnn._nbytes(t)
                else:
                    t_cpu = t.untyped_storage().cpu()
                    data_ptr = t_cpu.data_ptr()
                    nbytes = t_cpu.nbytes()

                raw_array = ctypes.cast(
                    data_ptr,
                    ctypes.POINTER(ctypes.c_ubyte * nbytes),
                )
                raw_bytes = bytes(raw_array.contents)
                return raw_bytes if all_cuda else _pad_to_alignment(raw_bytes)

            serialized_weights = b"".join(
                _to_bytes(graph.get_original_value_of_constant(name), all_cuda)
                for name in graph.constants.keys()
                if name not in graph.folded_constants
            )
            if not use_mmap_weights:
                aot_constants = serialized_weights
                magic_number = 0
            else:
                magic_number = cast(
                    int, torch.randint(0, torch.iinfo(torch.int64).max, (1,)).item()
                )
                aot_constants = struct.pack("qq", consts_size + 8, magic_number)

            consts_o = {
                "linux": _compile_consts_linux,
            }[
                sys.platform
            ](aot_constants)

            if config.aot_inductor.package:
                output_name, output_dir = get_name_and_dir_from_output_file_path(
                    output_so
                )
                so_build_options = CppTorchMusaOptions(
                    vec_isa=picked_vec_isa,
                    musa=musa,
                    aot_mode=graph.aot_mode,
                    use_absolute_path=use_absolute_path,
                )
                so_builder = CppBuilder(
                    name=output_name,
                    sources=[output_o, consts_o],
                    output_dir=output_dir,
                    BuildOption=so_build_options,
                )
                link_cmd = so_builder.get_command_line()
                output_so = so_builder.get_target_file_path()

                linker_flags = os.path.splitext(input_path)[0] + "_linker_flags.json"
                so_build_options.save_flags_to_file(linker_flags)

                if use_mmap_weights:
                    weight_file = (
                        os.path.splitext(input_path)[0] + "_serialized_weights.bin"
                    )
                    with open(weight_file, "wb") as f_weights:
                        f_weights.write(serialized_weights)
                        f_weights.write(struct.pack("q", magic_number))

                archive_path = package_aoti(os.path.split(input_path)[0])
                return archive_path
            output_name, output_dir = get_name_and_dir_from_output_file_path(output_so)
            so_build_options = CppTorchMusaOptions(
                vec_isa=picked_vec_isa,
                musa=musa,
                aot_mode=graph.aot_mode,
                use_absolute_path=use_absolute_path,
            )
            so_builder = CppBuilder(
                name=output_name,
                sources=[output_o, consts_o],
                output_dir=output_dir,
                BuildOption=so_build_options,
            )
            link_cmd = so_builder.get_command_line()
            output_so = so_builder.get_target_file_path()

            log.debug("aot linkage command: %s", link_cmd)
            if fbcode_aot_cpu_re:
                output_so = (
                    config.aot_inductor.output_path
                    if specified_so_name
                    else os.path.splitext(input_path)[0] + ".so"
                )
                compile_file([output_o, consts_o], output_so, link_cmd.split())
                os.chmod(output_so, 0o755)
            else:
                run_command_and_check(link_cmd)

            if use_mmap_weights:

                page_size_ = resource.getpagesize()
                page_size = max(16384, page_size_)

                with open(output_so, "a+b") as f_so:
                    so_size = f_so.tell()
                    # Page align the weights
                    f_so.write(b" " * (page_size - so_size % page_size))
                    f_so.write(serialized_weights)
                    f_so.write(struct.pack("q", magic_number))

            # Append cmds to the end of codegen-ed wrapper file
            with open(input_path, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write(f"// Compile cmd\n// {compile_cmd}\n")
                f.write(f"// Link cmd\n// {link_cmd}\n")

        return output_so


torch._inductor.codecache.get_hash = get_hash
torch._inductor.codecache.get_cpp_wrapper_cubin_path_name = (
    get_cpp_wrapper_mubin_path_name
)
