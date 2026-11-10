"""The setup file"""

# pylint: disable=W4901

import os
import glob
import sys
import sysconfig
import multiprocessing
import subprocess
import distutils.command.clean

from os.path import dirname, join
from pathlib import Path
from setuptools import Distribution, setup, find_packages
import setuptools.command.install
from setuptools.command.bdist_wheel import bdist_wheel as setuptools_bdist_wheel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(BASE_DIR, "tools", "cuda_porting")):
    BASE_DIR = os.getcwd()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CLEAN_MODE = "clean" in sys.argv
METADATA_MODE = any(arg in {"egg_info", "dist_info"} for arg in sys.argv)
# Building must not load an already-installed torch_musa backend: the in-tree
# bindings may be absent (BUILD_PYTHON=OFF) or stale.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
if not CLEAN_MODE:
    from torch.utils.cpp_extension import CppExtension  # pylint: disable=C0411
    from torch.utils.cpp_extension import (
        BuildExtension as Build,
    )  # pylint: disable=C0411


if os.getenv("MAX_JOBS") is None:
    os.environ["MAX_JOBS"] = str(multiprocessing.cpu_count())
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

RERUN_CMAKE = False


def build_python_enabled():
    """Return whether the native Python bindings should be built.

    Keep the default aligned with the CMake option in this repository while
    accepting the common CMake/Python spellings used by an upper-level build.
    """
    value = os.getenv("BUILD_PYTHON", "ON").strip().upper()
    if value in {"ON", "1", "YES", "TRUE", "Y", "T"}:
        return True
    if value in {"", "OFF", "0", "NO", "FALSE", "N", "F"}:
        return False
    raise ValueError(
        f"Invalid BUILD_PYTHON={value!r}; use ON/OFF, TRUE/FALSE, 1/0, or T/F"
    )


# Match the upstream libtorch-only wheel: a distinct distribution name so a
# no-Python install cannot overwrite a Python-enabled torch_musa.
PACKAGE_NAME = os.getenv(
    "TORCH_MUSA_PACKAGE_NAME",
    "torch_musa" if build_python_enabled() else "torch_musa_no_python",
)


class Install(setuptools.command.install.install):
    """
    Install class.
    """


class BinaryDistribution(Distribution):
    # Native libraries are built by CMake even when ext_modules is empty.
    # Match PyTorch's binary wheel layout and platform tagging.
    def has_ext_modules(self):
        return True


class bdist_wheel(setuptools_bdist_wheel):  # pylint: disable=invalid-name
    """Wheel builder that strips Python sources from the no-python wheel."""

    def write_wheelfile(self, *args, **kwargs):
        super().write_wheelfile(*args, **kwargs)
        # editable_wheel also calls write_wheelfile to create the wheel
        # metadata, but it does not populate bdist_dir with a staged package
        # tree. Stripping sources and creating an empty __init__.py there
        # would leave the editable install inconsistent.
        if build_python_enabled() or "editable_wheel" in sys.argv:
            return
        # Match the upstream libtorch-only wheel: strip the Python sources,
        # keep the native libraries, and leave an empty package marker.
        if self.bdist_dir is None:
            raise AssertionError("self.bdist_dir must not be None")
        bdist_dir = Path(self.bdist_dir)
        for path in bdist_dir.rglob("*.py"):
            path.unlink()
        (bdist_dir / "torch_musa" / "__init__.py").touch()


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


def build_musa_lib():
    """
    Build the MUSA native libraries.
    """
    tools_dir = os.path.join(BASE_DIR, "tools")
    tools_mod = sys.modules.get("tools")
    tools_path = getattr(tools_mod, "__file__", "") if tools_mod else ""
    if tools_mod and not os.path.abspath(tools_path).startswith(tools_dir):
        for name in list(sys.modules):
            if name == "tools" or name.startswith("tools."):
                del sys.modules[name]
    import types  # pylint: disable=import-outside-toplevel

    tools_pkg = types.ModuleType("tools")
    tools_pkg.__path__ = [tools_dir]
    tools_pkg.__file__ = os.path.join(tools_dir, "__init__.py")
    sys.modules["tools"] = tools_pkg
    from tools.cuda_porting.cuda_porting import (  # pylint: disable=import-outside-toplevel
        port_cuda,
    )

    # generate code for CUDA porting
    build_dir = "build"
    gen_porting_dir = "generated_cuda_compatible"
    code_generated_dir = "torch_musa_codegen"
    cuda_compatiable_path = os.path.join(BASE_DIR, build_dir, gen_porting_dir)
    code_generated_path = os.path.join(BASE_DIR, build_dir, code_generated_dir)
    if not os.path.isdir(cuda_compatiable_path):
        port_cuda(pytorch_root, get_pytorch_install_path(), cuda_compatiable_path)

    cmake = CMake(build_dir, install_dir_prefix="torch_musa")
    # use reference to os.environ first, because newly added env vars may be accessed in CMake.
    env = os.environ
    env["GENERATED_PORTING_DIR"] = cuda_compatiable_path
    env["CODE_GENERATED_DIR"] = code_generated_path
    # add `BUILD` prefix to avoid env being filtered.
    env["BUILD_PYTORCH_REPO_PATH"] = env["PYTORCH_REPO_PATH"]
    # Find Torch through its installed CMake package, including editable builds.
    env["CMAKE_PREFIX_PATH"] = ";".join(
        filter(None, [get_pytorch_install_path(), env.get("CMAKE_PREFIX_PATH")])
    )
    build_test = not check_negative_env_flag("BUILD_TEST")
    cmake_python_library = (
        f"{sysconfig.get_config_var('LIBDIR')}/{sysconfig.get_config_var('INSTSONAME')}"
    )
    # NOTE: `version` passed to cmake.generate won't take effect.
    # Because CMakeLists.txt will read the `version.txt` itself.
    cmake.generate(
        version,
        cmake_python_library,
        build_python_enabled(),
        build_test,
        env.copy(),
        RERUN_CMAKE,
    )
    cmake.build(env)


if not CLEAN_MODE and not METADATA_MODE:
    build_musa_lib()


def configure_extension_build():
    """
    Config the extension.
    """
    if CLEAN_MODE:
        return None
    if not build_python_enabled():
        # BUILD_PYTHON=OFF builds the native libraries only, mirroring the
        # libtorch-only build upstream. Do not emit _MUSAC/_ext, which would
        # link against the Python bindings that this configuration omits.
        return []
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

    python_libraries = ["musa_python"]
    torch_musa_sources = glob.glob("torch_musa/csrc/stub.cpp")
    cpp_extension = CppExtension(
        name="torch_musa._MUSAC",
        sources=torch_musa_sources,
        libraries=python_libraries + ["musa_kernels", "torch_musa"],
        include_dirs=[],
        extra_compile_args=extra_compile_args,
        library_dirs=[os.path.join(BASE_DIR, "torch_musa/lib")],
        extra_link_args=extra_link_args + ["-Wl,-rpath,$ORIGIN/lib"],
    )
    ext_extension = CppExtension(
        name="torch_musa._ext",
        sources=glob.glob("torch_musa/csrc/extension/C_frontend.cpp"),
        libraries=python_libraries
        + ["_ext_musa_kernels", "musa_kernels", "torch_musa"],
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

    cmdclass = {"clean": Clean, "install": Install, "bdist_wheel": bdist_wheel}
    if not CLEAN_MODE:
        cmdclass["build_ext"] = Build

    setup(
        name=PACKAGE_NAME,
        distclass=BinaryDistribution,
        version=version,
        description="A PyTorch backend extension for Moore Threads MUSA",
        url="https://github.com/MooreThreads/torch_musa",
        author="Moore Threads PyTorch AI Dev Team",
        packages=find_packages(exclude=["tools", "tools*"]),
        ext_modules=configure_extension_build(),
        include_package_data=True,
        install_requires=install_requires,
        extras_require={},
        entry_points=(
            {
                "console_scripts": [
                    "musa-converter = torch_musa.utils.musa_converter:main"
                ],
                "torch.backends": ["torch_musa = torch_musa:_autoload"],
            }
            if build_python_enabled()
            else {}
        ),
        cmdclass=cmdclass,
    )
