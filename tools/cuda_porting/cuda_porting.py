import os
import shutil
import ahocorasick
from typing import Dict
from tools.cuda_porting.match_rewrite import init_ac_automaton, transform_file

class PortingFile:
    r"""Class used to manage porting files.
    """

    def __init__(self, dir_name: str, recursive: bool, need_filter_cpp: bool) -> None:
        r"""Initializes class.

        Args:
            dir_name (str): folder name of porting files.
            recursive (bool): whether to port recursively in the folder or not.
            need_filter_cpp (bool): whether to filter out cpp files or not.
        """
        self.dir_name = dir_name
        self.recursive = recursive
        self.need_filter_cpp = need_filter_cpp

r"""All folders needed for cuda-porting
"""
PORT_FILES = [PortingFile("aten/src/ATen/native", True, False),
              PortingFile("aten/src/ATen/cuda", True, False),
              PortingFile("aten/src/ATen/core", True, True),
              PortingFile("aten/src/ATen/detail", True, True),
              PortingFile("aten/src/ATen/", False, True),
              PortingFile("c10/core", True, True),
              PortingFile("c10/cuda", True, False),
              PortingFile("c10/macros", True, True),
              PortingFile("c10/util", True, True),
              PortingFile("include/ATen/ops", True, True),
              PortingFile("include/ATen/core", False, True),
              PortingFile("include/ATen", False, True)]

def get_automaton(input_map: Dict[str, str]) -> ahocorasick.Automaton:
    r"""Get an instance of Automaton for matching and replacing.

    Args:
        input_map (Dict[str, str]): description of source string and destination string in Automaton.

    Returns:
        Automaton: an instance of Automaton with input_map.
    """

    automaton = ahocorasick.Automaton()
    for cuda, musa in input_map.items():
        automaton.add_word(cuda, (len(cuda), musa))
    automaton.make_automaton()
    return automaton

def port_cuda(pytorch_src_root: str, pytorch_install_root: str, generated_dir: str) -> None:
    r"""Port files for cuda compatibility.
    
    Args:
        pytorch_src_root (str): path of PyTorch source code.
        pytorch_install_root (str): path of PyTorch installed.
        generated_dir (str): path of which porting files will be generated.

    Returns: None.
    """
    # Prepare map used to replace the header file path
    split_generated_path = generated_dir.split('/')
    relative_header_file_path = split_generated_path[-2] + '/' + split_generated_path[-1]
    headers_path_map = {"#include <ATen": "#include <" + relative_header_file_path + "/aten/src/ATen",
                        "#include <c10": "#include <" + relative_header_file_path + "/c10"}
    headers_automaton = get_automaton(headers_path_map)

    # Prepare main cuda-porting map
    current_dir = os.path.dirname(os.path.abspath(__file__))
    map_files = [os.path.join(current_dir, "include.json"),
                 os.path.join(current_dir, "general.json"),
                 os.path.join(current_dir, "extra.json")]
    main_automaton = init_ac_automaton(map_files, headers_path_map)

    # Prepare disable half dtype map
    # TODO(caizhi): the following map is used to disable half dtype in operations.
    # It will be deleted when mcc compiler supports half dtype.
    disable_half_map = {"AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)": "",
                        "AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)": "",
                        "AT_DISPATCH_FLOATING_TYPES_AND_HALF": "AT_DISPATCH_FLOATING_TYPES",
                        "AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half,": "AT_DISPATCH_ALL_TYPES(",
                        "AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf,": "AT_DISPATCH_COMPLEX_TYPES("}
    disable_half_automaton = get_automaton(disable_half_map)

    # Prepare replace map which is not handled in cuda-porting tools
    extra_replace_map = {"cudaOccupancy": "musaOccupancy",
                         "set_cuda_dispatch_ptr(value)": "set_musa_dispatch_ptr(value)",
                         "namespace cuda {": "namespace musa {"}

    # 1. Copy and cuda-port files
    for port_file in PORT_FILES:
        src_root = pytorch_install_root if "include" in port_file.dir_name else pytorch_src_root
        for cur_dir, sub_dir, files in os.walk(os.path.join(src_root, port_file.dir_name)):
            relative_path = cur_dir.replace(src_root.rstrip('/') + "/", "")
            destination_folder = os.path.join(generated_dir, relative_path.replace("cuda", "musa").replace("include","aten/src"))

            if not port_file.recursive:
                if relative_path != port_file.dir_name:
                    continue

            if not os.path.isdir(destination_folder):
                os.makedirs(destination_folder)

            for f in files:
                file_path = os.path.join(cur_dir, f)
                if port_file.need_filter_cpp:
                    if ".h" in file_path or ".cu" in file_path:
                        shutil.copy(file_path, destination_folder)
                    else:
                        continue
                else:
                    shutil.copy(file_path, destination_folder)

                dst_file = os.path.join(destination_folder, f)
                dst_file = transform_file(dst_file, main_automaton, extra_replace_map)
                transform_file(dst_file, disable_half_automaton)

    # 2. Copy several special files about macros files
    special_copy_files = {"cmake_macros.h": "c10/macros",
                          "cuda_cmake_macros.h": "c10/musa/impl",
                          "CUDAConfig.h": "aten/src/ATen/musa",
                          "jit_macros.h": "aten/src/ATen"}
    for key, value in special_copy_files.items():
        shutil.copy(os.path.join(current_dir, key), os.path.join(generated_dir, value))

