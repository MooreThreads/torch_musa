# pylint: skip-file
"""AOTI compiler"""

from typing import Dict, Optional, Union, cast, Any
import os
import sys
import json
import shutil
import dataclasses
from pathlib import Path
import struct
from filelock import FileLock

import torch
from torch._inductor import config
from torch._inductor.utils import ALIGN_BYTES
from torch._inductor.cpp_builder import (
    CppBuilder,
    get_name_and_dir_from_output_file_path,
    _LINKER_SCRIPT,
)
from torch.utils._ordered_set import OrderedSet
from torch._inductor.utils import (
    clear_on_fresh_inductor_cache,
    is_linux,
)
from torch._inductor.codecache import (
    split_aot_inductor_output_path,
    code_hash,
    write,
    write_atomic,
    pick_vec_isa,
    get_lock_dir,
    LOCK_TIMEOUT,
    output_code_log,
    log,
    DLLWrapper,
    CUDACodeCache,
)
from torch._inductor.package.pt2_archive_constants import CUSTOM_OBJ_FILENAME_PREFIX
from torch_musa.utils.musa_extension import _find_musa_home
from ..cpp_builder import CppTorchMusaOptions


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
        wrapper_code: str,
        kernel_code: str,
        serialized_extern_kernel_nodes: Optional[str],
        *,
        device_type: str,
        additional_files: list[str],
    ) -> Union[list[str], str]:
        """
        compile generated MUSA C++ code to .so.

        graph: GraphLowering class
        source_code: generated MUSA C++ code.
        serialized_extern_kernel_nodes: triton kernel binary file.
        musa: if is musa device
        """
        musa = device_type == "musa"
        generated_files = additional_files

        _find_musa_home()  # cpp_extension consults the env

        picked_vec_isa = pick_vec_isa()
        vec_isa_cmd_gen = CppBuilder(
            name="o",
            sources="i",
            BuildOption=CppTorchMusaOptions(
                vec_isa=picked_vec_isa,
                device_type=device_type,
                aot_mode=graph.aot_mode,
            ),
        )
        cpp_command = repr(vec_isa_cmd_gen.get_command_line())

        use_relative_path = False

        ld_command = "ld"
        objcopy_command = "objcopy"

        (
            specified_output_path,
            specified_so_name,
        ) = split_aot_inductor_output_path(config.aot_inductor.output_path)
        if config.aot_inductor.package_cpp_only:
            wrapper_code = "\n".join((wrapper_code, kernel_code))
            kernel_code = ""

        wrapper_key, wrapper_path = write(
            wrapper_code,
            "wrapper.cpp",
            extra=cpp_command,
            specified_dir=specified_output_path,
        )
        _, kernel_path = write(
            kernel_code,
            "kernel.cpp",
            extra=cpp_command,
            specified_dir=specified_output_path,
        )

        if config.aot_inductor.package:
            generated_files.append(wrapper_path)
            if not config.aot_inductor.package_cpp_only:
                generated_files.append(kernel_path)

        output_code_log.info("Output code written to: %s", wrapper_path)
        output_code_log.info("Kernel code written to : %s", kernel_path)
        torch._logging.trace_structured(
            "graph_dump",
            lambda: {
                "name": "inductor_aot_wrapper_code",
                "type": "cpp",
                "filename": wrapper_path,
            },
            payload_fn=lambda: wrapper_code,
        )
        torch._logging.trace_structured(
            "graph_dump",
            lambda: {
                "name": "inductor_aot_kernel_code",
                "type": "cpp",
                "filename": kernel_path,
            },
            payload_fn=lambda: kernel_code,
        )

        wrapper_path_operator = Path(wrapper_path)
        kernel_path_operator = Path(kernel_path)
        specified_sub_dir = wrapper_path_operator.parent / wrapper_key
        if not specified_sub_dir.exists():
            specified_sub_dir.mkdir(exist_ok=True)
        cmake_path = str(Path(specified_sub_dir) / "CMakeLists.txt")

        def _compile_consts_linux(consts: bytes, platform: str) -> str:
            if platform == "linux":
                if graph.mutated_buffers & OrderedSet(graph.constants.keys()):
                    if len(consts) > 2_000_000_000:
                        raise ValueError(
                            "Models with buffer mutation included doesn't support constants greater than 2GB!"
                        )
                    section_attr = '.ldata, "aw"'
                else:
                    section_attr = '.lrodata, "a"'
                symbol_prefix = ""
            is_large_consts = len(consts) > 1024
            consts_asm = f"\t.section\t{section_attr}\n"
            consts_asm += f"\t.balign {ALIGN_BYTES}\n"
            consts_asm += f"\t.globl\t{symbol_prefix}_binary_constants_bin_start\n"
            consts_asm += f"{symbol_prefix}_binary_constants_bin_start:\n"
            if not is_large_consts:
                for c in consts:
                    consts_asm += f"\t.byte {c}\n"
                # Add one element even if constants are empty
                # Otherwise assembler will not put them in data section
                if not consts:
                    consts_asm += "\t.space 1\n"
            else:
                consts_asm += "\t.quad 0x1234567899abcdef\n"
                consts_asm += f"\t.space {len(consts) - 8}\n"
            consts_asm += f".globl\t{symbol_prefix}_binary_constants_bin_end\n"
            consts_asm += f"{symbol_prefix}_binary_constants_bin_end:\n"
            _, consts_path = write(
                consts_asm,
                "S",
                specified_dir=str(specified_sub_dir),
            )

            consts_s = Path(consts_path)
            object_build_options = CppTorchMusaOptions(
                device_type=device_type,
                aot_mode=graph.aot_mode,
                compile_only=True,
                use_relative_path=use_relative_path,
            )

            object_builder = CppBuilder(
                name=str(consts_s.stem),
                sources=str(consts_s),
                output_dir=str(consts_s.parent),
                BuildOption=object_build_options,
            )
            consts_o = object_builder.get_target_file_path()
            object_builder.build()

            if is_large_consts:
                with open(consts_o, "r+b") as f:
                    f.seek(0)
                    hdr = f.read(1024)
                    # Search for magic number and write the actual data over it
                    start_idx = hdr.find(b"\xef\xcd\xab\x99\x78\x56\x34\x12")
                    assert start_idx != -1
                    f.seek(start_idx)
                    pos = 0
                    while pos < len(consts):
                        rc = f.write(consts[pos:])
                        pos += rc

            os.remove(consts_s)

            return consts_o

        lock_dir = get_lock_dir()
        lock = FileLock(
            os.path.join(lock_dir, wrapper_key + ".lock"), timeout=LOCK_TIMEOUT
        )
        with lock:
            if serialized_extern_kernel_nodes:
                extern_kernel_nodes_json = str(
                    wrapper_path_operator.with_suffix(".json")
                )
                with open(extern_kernel_nodes_json, "w") as f:
                    f.write(serialized_extern_kernel_nodes)

                if config.aot_inductor.package:
                    generated_files.append(extern_kernel_nodes_json)
            # Save user provided metadata
            meta_json = str(
                wrapper_path_operator.with_name(
                    f"{wrapper_path_operator.stem}_metadata.json"
                )
            )
            for k, v in config.aot_inductor.metadata.items():
                assert isinstance(k, str) and isinstance(
                    v, (str)
                ), "Metadata must only contain strings"

            with open(meta_json, "w") as f:
                f.write(json.dumps(config.aot_inductor.metadata))

            kernel_meta_json = str(
                kernel_path_operator.with_name(
                    f"{kernel_path_operator.stem}_metadata.json"
                )
            )
            shutil.copy(meta_json, kernel_meta_json)

            if config.aot_inductor.package:
                generated_files.append(meta_json)
                if not config.aot_inductor.package_cpp_only:
                    generated_files.append(kernel_meta_json)
            output_so = (
                config.aot_inductor.output_path
                if specified_so_name
                else str(wrapper_path_operator.with_suffix(".so"))
            )

            all_musa = all(
                graph.get_original_value_of_constant(name).is_musa
                for name in graph.constants.keys()
                if name not in graph.folded_constants
            )

            def _to_bytes(t: torch.Tensor, all_musa: bool) -> bytes:
                def _pad_to_alignment(raw_bytes: bytes) -> bytes:
                    padded_bytes = raw_bytes.ljust(
                        (len(raw_bytes) + ALIGN_BYTES - 1) // ALIGN_BYTES * ALIGN_BYTES,
                        b"\x00",
                    )
                    return padded_bytes

                # This serializes the tensor's untyped_storage to bytes by accessing
                # the raw data of the underlying structure.
                import ctypes

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
                return raw_bytes if all_musa else _pad_to_alignment(raw_bytes)

            if config.aot_inductor.package_constants_in_so:
                serialized_weights = b"".join(
                    _to_bytes(graph.get_original_value_of_constant(name), all_musa)
                    for name in graph.constants.keys()
                    if name not in graph.folded_constants
                )
            else:
                serialized_weights = b""

            consts_size = len(serialized_weights)
            # TODO: Fix mmap weights with cuda
            use_mmap_weights = not config.is_fbcode() and consts_size > 2_000_000_000
            if config.aot_inductor.force_mmap_weights:
                use_mmap_weights = True

            compile_command: dict[str, Any] = {
                "aot_mode": graph.aot_mode,
                "device_type": device_type,
                "use_mmap_weights": use_mmap_weights,
                "use_relative_path": config.is_fbcode(),
                "vec_isa": picked_vec_isa,
            }

            wrapper_build_options = CppTorchMusaOptions(
                compile_only=True,
                min_optimize=not config.aot_inductor.package_cpp_only,
                **compile_command,
            )
            kernel_build_options = CppTorchMusaOptions(
                compile_only=True,
                **compile_command,
            )

            wrapper_builder = CppBuilder(
                name=str(wrapper_path_operator.stem),
                sources=wrapper_path,
                output_dir=str(wrapper_path_operator.parent),
                BuildOption=wrapper_build_options,
            )

            wrapper_compile_cmd = wrapper_builder.get_command_line()
            wrapper_o = wrapper_builder.get_target_file_path()

            kernel_builder = CppBuilder(
                name=str(kernel_path_operator.stem),
                sources=kernel_path,
                output_dir=str(wrapper_path_operator.parent),
                BuildOption=kernel_build_options,
            )
            kernel_compile_cmd = kernel_builder.get_command_line()
            kernel_o = kernel_builder.get_target_file_path()

            log.debug("aot wrapper compilation command: %s", wrapper_compile_cmd)
            log.debug("aot kernel compilation command: %s", kernel_compile_cmd)

            if config.aot_inductor.package_cpp_only:
                # Not doing the actual compilation here
                compile_flags = str(
                    wrapper_path_operator.with_name(
                        f"{wrapper_path_operator.stem}_compile_flags.json"
                    )
                )
                wrapper_build_options.save_flags_to_json(compile_flags)
                generated_files.append(compile_flags)
                wrapper_builder.save_compile_cmd_to_cmake(print())
                wrapper_builder.save_src_to_cmake(cmake_path, wrapper_path)
                generated_files.append(cmake_path)
            else:
                wrapper_builder.build()
                kernel_builder.build()

            if not use_mmap_weights:
                aot_constants = serialized_weights
                magic_number = 0
            else:
                magic_number = cast(
                    int, torch.randint(0, torch.iinfo(torch.int64).max, (1,)).item()
                )
                aot_constants = struct.pack("qq", consts_size + 8, magic_number)

            consts_o = _compile_consts_linux(aot_constants, sys.platform)
            custom_obj_idx = 0
            # Note that custom_objs_config.json file is different from the model_constants_config.json file produced
            # in package_sigmoid(). The keys in custom_objs_config.json directly correspond to the arg name in extern
            # nodes json. The key in model_constants_config.json produced by package_sigmoid is the attribute name in the
            # user model code.

            qual_name_to_id = (
                {}
            )  # Map from constant name to its name in constants folder
            for custom_obj_idx, (name, constant) in enumerate(
                graph.torchbind_constants.items()
            ):
                assert isinstance(constant, torch._C.ScriptObject)
                custom_obj_name = f"{CUSTOM_OBJ_FILENAME_PREFIX}{custom_obj_idx}"

                log.debug("saving script object %s as %s", name, custom_obj_name)

                qual_name_to_id[name] = custom_obj_name
                custom_obj_bytes = torch._C._pickle_save(constant)
                custom_obj_path = os.path.join(
                    wrapper_path_operator.parent, custom_obj_name
                )

                write_atomic(custom_obj_path, custom_obj_bytes, True)
                generated_files.append(custom_obj_path)

            constants_config_json = os.path.join(
                wrapper_path_operator.parent, "custom_objs_config.json"
            )
            with open(constants_config_json, "w") as f:
                f.write(json.dumps(qual_name_to_id))
            generated_files.append(constants_config_json)

            gpu_codecache = CUDACodeCache()

            gpu_kernels_o = [
                entry.output_path
                for entry in gpu_codecache.cache.values()
                if entry.output_path.endswith(".o")
            ]
            gpu_kernels_o = " ".join(gpu_kernels_o)
            output_name, output_dir = get_name_and_dir_from_output_file_path(output_so)

            so_build_options = CppTorchMusaOptions(
                vec_isa=picked_vec_isa,
                device_type=device_type,
                aot_mode=graph.aot_mode,
                use_relative_path=False,
            )

            so_builder = CppBuilder(
                name=output_name,
                sources=(
                    [wrapper_o, kernel_o, consts_o, gpu_kernels_o]
                    if gpu_kernels_o
                    else [wrapper_o, kernel_o, consts_o]
                ),
                output_dir=output_dir,
                BuildOption=so_build_options,
            )

            link_cmd = so_builder.get_command_line()
            output_so = so_builder.get_target_file_path()

            log.debug("aot linkage command: %s", link_cmd)

            # Append cmds to the end of codegen-ed wrapper file
            with open(wrapper_path, "a") as f:
                f.write("\n")
                f.write(f"// Compile cmd\n// {wrapper_compile_cmd}\n")
                f.write(f"// Link cmd\n// {link_cmd}\n")

            with open(kernel_path, "a") as f:
                f.write("\n")
                f.write(f"// Compile cmd\n// {kernel_compile_cmd}\n")
                f.write(f"// Link cmd\n// {link_cmd}\n")

            if config.aot_inductor.package_cpp_only:
                linker_flags = str(
                    wrapper_path_operator.with_name(
                        f"{wrapper_path_operator.stem}_linker_flags.json"
                    )
                )
                so_build_options.save_flags_to_json(linker_flags)
                generated_files.append(linker_flags)
                generated_files.append(_LINKER_SCRIPT)

                # If we only want to package the cpp, then we need to save the
                # weights separately into a bin, and we also need to prevent compiling the so

                if use_mmap_weights:
                    weight_file = str(
                        wrapper_path_operator.with_name(
                            f"{wrapper_path_operator.stem}_serialized_weights.bin"
                        )
                    )
                    with open(weight_file, "wb") as f_weights:
                        f_weights.write(serialized_weights)
                        f_weights.write(struct.pack("q", magic_number))

                    generated_files.append(weight_file)

                generated_files.append(consts_o)
                generated_files.append(gpu_kernels_o)

                so_builder.save_src_to_cmake(cmake_path, consts_o)
                for gpu_o in gpu_kernels_o.split():
                    so_builder.save_src_to_cmake(cmake_path, gpu_o)
                so_builder.save_link_cmd_to_cmake(cmake_path)
            else:
                so_builder.build()

                # for o_file in [wrapper_o, kernel_o, consts_o]:
                #     # Remove these as they are not needed anymore
                #     os.remove(o_file)

                if use_mmap_weights:
                    import resource

                    page_size_ = resource.getpagesize()
                    page_size = max(16384, page_size_)

                    with open(output_so, "a+b") as f_so:
                        so_size = f_so.tell()
                        # Page align the weights
                        f_so.write(b" " * (page_size - so_size % page_size))
                        f_so.write(serialized_weights)
                        f_so.write(struct.pack("q", magic_number))

                if config.aot_inductor.package:
                    generated_files.append(output_so)

        if config.aot_inductor.package:
            # We want to return the directory that contains all the AOTI
            # generated files, not just the so
            # return os.path.split(output_so)[0]
            return generated_files

        return output_so


torch._inductor.codecache.get_hash = get_hash
torch._inductor.codecache.get_cpp_wrapper_cubin_path_name = (
    get_cpp_wrapper_mubin_path_name
)
torch._inductor.codecache.AotCodeCompiler = AotCodeCompiler
