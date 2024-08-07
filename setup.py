#
"""The setup file"""

import os
import glob
import sys
import sysconfig
import multiprocessing
import subprocess
import distutils.command.clean

from os.path import dirname, join
from subprocess import PIPE, Popen
from setuptools import setup, find_packages
import setuptools.command.install

from torch.utils.cpp_extension import CppExtension  # pylint: disable=C0411
from torch.utils.cpp_extension import BuildExtension as Build  # pylint: disable=C0411
from tools.cuda_porting.cuda_porting import port_cuda


if os.getenv("MAX_JOBS") is None:
    os.environ["MAX_JOBS"] = str(multiprocessing.cpu_count())

CLEAN_MODE = False
for i, arg in enumerate(sys.argv):
    if arg == "clean":
        CLEAN_MODE = True
if not CLEAN_MODE:
    pytorch_root = os.getenv("PYTORCH_REPO_PATH", default="")
    if pytorch_root == "":
        raise RuntimeError(
            "Building error: PYTORCH_REPO_PATH must be set to"
            " PyTorch repository when building, but now it is empty!"
        )
    sys.path.append(join(dirname(__file__), "torch_musa"))
    from setup_helpers.env import check_negative_env_flag, build_type
    from setup_helpers.cmake import CMake

with open("version.txt", "r", encoding="utf-8") as version_file:
    version = version_file.readlines()[0].strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RERUN_CMAKE = False


class Install(setuptools.command.install.install):
    """
    Install class.
    """


class Clean(distutils.command.clean.clean):
    """
    Clean class.
    """

    def run(self):
        # pylint: disable=import-outside-toplevel
        import re
        import shutil

        with open(".gitignore", "r", encoding="utf-8") as f:
            ignores = f.read()
            pat = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
            for wildcard in filter(None, ignores.split("\n")):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


def get_pytorch_install_path():
    """
    Get Pytorch installed path.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import torch

        pytorch_install_root = os.path.dirname(os.path.abspath(torch.__file__))
    except Exception as exe:
        raise RuntimeError(
            "Building error: import torch failed when building!"
        ) from exe
    return pytorch_install_root


def get_mtgpu_arch():
    """
    Get moorethreads gpu arch.
    """
    mthreads_gmi = "mthreads-gmi"

    # name and arch dict
    name_arches = {
        "MTT S2000": "11",
        "MTT S3000": "21",
        "MTT S80": "21",
        "MTT S80ES": "21",
        "MTT S4000": "22",
        "MTT S90": "22",
    }

    # Get ID, processing and memory utilization for all GPUs
    try:
        with Popen([mthreads_gmi, "-q -i 0"], stdout=PIPE) as p:
            stdout, _ = p.communicate()
    except Exception as exception:
        raise RuntimeError("Unable to run the mthreads-gmi command") from exception
    output = stdout.decode("UTF-8")

    lines = output.split(os.linesep)

    for line in lines:
        kvs = line.split(" : ")
        if len(kvs) != 2:
            continue
        if kvs[0].strip().startswith("Product Name"):
            name = kvs[1].strip()
            return name_arches[name]
    raise RuntimeError(
        "Can not find Product Name in the output of 'mthreads-gmi -q -i 0'"
    )


def build_musa_lib():
    """
    Build musa python lib.
    """
    # generate code for CUDA porting
    build_dir = "build"
    gen_porting_dir = "generated_cuda_compatible"
    code_generated_dir = "torch_musa_codegen"
    cuda_compatiable_path = os.path.join(BASE_DIR, build_dir, gen_porting_dir)
    code_generated_path = os.path.join(BASE_DIR, build_dir, code_generated_dir)
    if not os.path.isdir(cuda_compatiable_path):
        port_cuda(pytorch_root, get_pytorch_install_path(), cuda_compatiable_path)

    os.environ["MUSA_ARCH"] = get_mtgpu_arch()

    cmake = CMake(build_dir, install_dir_prefix="torch_musa")
    # use reference to os.environ first, because newly added env vars may be accessed in CMake.
    env = os.environ
    env["GENERATED_PORTING_DIR"] = cuda_compatiable_path
    env["CODE_GENERATED_DIR"] = code_generated_path
    # add `BUILD` prefix to avoid env being filtered.
    env["BUILD_PYTORCH_REPO_PATH"] = env["PYTORCH_REPO_PATH"]
    build_test = not check_negative_env_flag("BUILD_TEST")
    cmake_python_library = (
        f"{sysconfig.get_config_var('LIBDIR')}/{sysconfig.get_config_var('INSTSONAME')}"
    )
    # NOTE: `version` passed to cmake.generate won't take effect.
    # Because CMakeLists.txt will read the `version.txt` itself.
    cmake.generate(
        version, cmake_python_library, True, build_test, env.copy(), RERUN_CMAKE
    )
    cmake.build(env)


if not CLEAN_MODE:
    build_musa_lib()


def configure_extension_build():
    """
    Config the extension.
    """
    if CLEAN_MODE:
        return None
    extra_link_args = []
    extra_compile_args = [
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fno-strict-aliasing",
        "-fstack-protector-all",
    ]

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

    torch_musa_sources = glob.glob("torch_musa/csrc/stub.cpp")
    cpp_extension = CppExtension(
        name="torch_musa._MUSAC",
        sources=torch_musa_sources,
        libraries=["musa_python"],
        include_dirs=[],
        extra_compile_args=extra_compile_args,
        library_dirs=[os.path.join(BASE_DIR, "torch_musa/lib")],
        extra_link_args=extra_link_args + ["-Wl,-rpath,$ORIGIN/lib"],
    )
    ext_extension = CppExtension(
        name="torch_musa._ext",
        sources=glob.glob("torch_musa/csrc/extension/C_frontend.cpp"),
        libraries=["_ext_musa_kernels", "musa_python"],
        include_dirs=[],
        extra_compile_args={"cxx": ["-std=c++17"]},
        library_dirs=[os.path.join(BASE_DIR, "torch_musa/lib")],
        extra_link_args=extra_link_args + ["-Wl,-rpath,$ORIGIN/lib"],
    )
    return [cpp_extension, ext_extension]


install_requires = ["packaging"]


def package_files(directory):
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            paths.append(os.path.join("..", root, file))
    return paths


def dump_version():
    """
    Dump the torch_musa version.
    """
    here = BASE_DIR
    version_file_path = os.path.join(here, "torch_musa", "version.py")
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here)
        .decode("ascii")
        .strip()
    )
    __version__ = version + f"+{sha[:7]}"
    with open(version_file_path, "w", encoding="utf-8") as f:
        f.write(f"__version__ = '{__version__}'\n")
        f.write(f"git_version = {repr(sha)}\n")


# Setup
if __name__ == "__main__":
    dump_version()

    setup(
        name="torch_musa",
        version=version,
        description="A PyTorch backend extension for Moore Threads MUSA",
        url="https://github.mthreads.com/mthreads/torch_musa",
        author="Moore Threads PyTorch AI Dev Team",
        packages=find_packages(exclude=["tools", "tools*"]),
        ext_modules=configure_extension_build(),
        include_package_data=True,
        install_requires=install_requires,
        extras_require={},
        entry_points={
            "console_scripts": ["musa-converter = torch_musa.utils.musa_converter:main"]
        },
        cmdclass={"build_ext": Build, "clean": Clean, "install": Install},
    )
