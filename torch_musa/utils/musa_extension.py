"""The musa extension file"""
import multiprocessing
import os
import sysconfig
from os.path import dirname, realpath, join

import torch
from torch.utils.cpp_extension import CppExtension

import torch_musa
from torch_musa.setup_helpers.cmake import CMake
from torch_musa.setup_helpers.env import build_type, check_negative_env_flag
from torch_musa.utils.cmake_lists_generator import CMakeListsGenerator

if os.getenv("MAX_JOBS") is None:
    os.environ["MAX_JOBS"] = str(multiprocessing.cpu_count())

RERUN_CMAKE = False
install_requires = ["packaging"]


# pylint: disable=C0103
def MUSAExtension(name, sources, *args, **kwargs):
    r"""The user interface of MUSAExtension is consistent with
    torch.utils.cpp_extension.CppExtension but some extra parameters can be recognized. note:
    1.`package_name` is obtained from `name`, if `name` is `example._C` then `package_name` is
    `example` 2.`project_name` is obtained from `package_name`, it is `EXAMPLE` in the above case
    but you can set it via MUSAExtension(name, sources, project_name=xxx) 3.`plugin_name` is
    obtained from `package_name`, it is `example` in the above case but you can set it via
    MUSAExtension(name, sources, plugin_name=xxx) 4.`project_name` and `plugin_name` will be
    filled in auto-generated CMakeLists.txt hence `libexample.so` will be born in the above case
    5.`package_name` is consistent with the name of directory which contains sources,
    like `example/csrc, example/core, example/testing` etc
    """
    version = "unknown"
    torch_musa_dir_path = realpath(dirname(dirname(dirname(__file__))))

    # split mu and cpp files
    mu_srcs = []
    cpp_srcs = []
    for src in sources:
        if src.endswith(".mu"):
            mu_srcs.append(src)
        else:
            cpp_srcs.append(src)

    # define include dirs
    include_dirs = [join(torch_musa_dir_path, path) for path in [
        "/usr/local/musa/include",
        "build/generated_cuda_compatible/aten/src",
        "build/generated_cuda_compatible/include",
        "build/generated_cuda_compatible/include/torch/csrc/api/include",
    ]] + [torch_musa_dir_path] + kwargs.get('include_dirs', [])
    kwargs['include_dirs'] = include_dirs

    # define library_dirs
    library_dirs = kwargs.get('library_dirs', [])
    kwargs['library_dirs'] = library_dirs

    # define libraries
    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    kwargs['libraries'] = libraries

    kwargs['language'] = 'c++'

    # utilize CMakeLists.txt to compile mu files
    musa_link_libraries = kwargs.get("musa_link_libraries", [])
    name_splits = name.split('.')

    if len(name_splits) > 1:
        package_name = name_splits[0]
        ext_name = "_".join(name_splits)
    elif len(name_splits) == 1:
        package_name = name
        ext_name = name
    else:
        raise RuntimeError(f"Error: got MUSAExtension name={name}")

    project_name = kwargs.get("project_name", package_name.upper())
    plugin_name = kwargs.get("plugin_name", ext_name)
    CMakeListsGenerator(sources=mu_srcs, include_dirs=include_dirs,
                        link_libraries=musa_link_libraries,
                        define_macros=kwargs.get("define_macros", []),
                        project_name=project_name,
                        plugin_name=plugin_name).generate()
    cmake = CMake(build_dir="build", install_dir_prefix=package_name)
    env = os.environ.copy()
    build_test = not check_negative_env_flag("BUILD_TEST")
    cmake_python_library = f"{sysconfig.get_config_var('LIBDIR')}/" \
                           f"{sysconfig.get_config_var('INSTSONAME')}"
    cmake.generate(version, cmake_python_library, True, build_test, env, RERUN_CMAKE)
    cmake.build(env)

    # define extra_compile_args
    user_extra_compile_args = kwargs.get('extra_compile_args', [])
    if isinstance(user_extra_compile_args, dict):
        user_extra_compile_args = []
    extra_compile_args = [
                             "-std=c++14",
                             "-Wall",
                             "-Wextra",
                             "-fno-strict-aliasing",
                             "-fstack-protector-all",
                         ] + user_extra_compile_args
    extra_link_args = []

    if build_type.is_debug():
        extra_compile_args += ["-O0", "-g"]
        extra_link_args += ["-O0", "-g"]

    if build_type.is_rel_with_deb_info():
        extra_compile_args += ["-g"]
        extra_link_args += ["-g"]

    use_asan = os.getenv("USE_ASAN", default="").upper() in [
        "ON",
        "1",
        "YES",
        "TRUE",
        "Y",
    ]

    if use_asan:
        extra_compile_args += ["-fsanitize=address"]
        extra_link_args += ["-fsanitize=address"]

    torch_lib_path = join(dirname(torch.__file__), "lib")
    torch_musa_lib_path = join(dirname(torch_musa.__file__), "lib")
    extra_link_args = extra_link_args + ["-Wl,-rpath,$ORIGIN/lib"] + \
                      [f"-Wl,-rpath,{torch_lib_path}"] + [f"-Wl,-rpath,{torch_musa_lib_path}"]

    library_dirs.append(f"{package_name}" + "/lib")
    libraries.append(f"{plugin_name}")

    kwargs['extra_compile_args'] = extra_compile_args
    kwargs['extra_link_args'] = extra_link_args
    return CppExtension(name, cpp_srcs, *args, **kwargs)
