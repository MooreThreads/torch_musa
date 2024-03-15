import sys
import os
import sysconfig
from typing import Optional, Dict, cast
from setup_helpers.env import IS_LINUX
from setup_helpers.numpy_ import USE_NUMPY, NUMPY_INCLUDE_DIR
from setup_helpers.cmake import CMake


def _mkdir_p(d: str) -> None:
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Failed to create folder {os.path.abspath(d)}: {e.strerror}"
        ) from e


class CMakeManager(CMake):
    """Manages cmake.

    Arguments:
        build_dir (str): Folder used for compiling.
    """

    def __init__(self, build_dir: str = "build") -> None:
        super(CMakeManager, self).__init__(build_dir)

    def generate(
        self,
        version: Optional[str],
        cmake_python_library: Optional[str],
        build_python: bool,
        build_test: bool,
        my_env: Dict[str, str],
        rerun: bool,
    ) -> None:
        """Runs cmake to generate native build files..

        Args:
            version : (str): Version of torch_musa.
            cmake_python_library (str): Path of libpython.so on the system.
            build_python (bool): Whether to build python.
            build_test (bool): Whether to build test.
            my_env (Dict[str, str]): Dictionary containing environment variables.
            rerun (bool): Whether to rerun cmake.

        Returns:
            None.

        """

        if rerun and os.path.isfile(self._cmake_cache_file):
            os.remove(self._cmake_cache_file)

        args = []
        if not IS_LINUX:
            raise Exception(
                "Building error: torch_musa could be built only on Linux now, please check your operation system!"
            )
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        install_dir = os.path.join(base_dir, "torch_musa")

        _mkdir_p(install_dir)
        _mkdir_p(self.build_dir)

        # Store build options that are directly stored in environment variables
        build_options: Dict[str, CMakeValue] = {}

        # Build options that do not start with "BUILD_", "USE_", or "CMAKE_" and are directly controlled by env vars.
        # This is a dict that maps environment variables to the corresponding variable name in CMake.
        additional_options = {
            # Key: environment variable name. Value: Corresponding variable name to be passed to CMake. If you are
            # adding a new build option to this block: Consider making these two names identical and adding this option
            # in the block below.
            "_GLIBCXX_USE_CXX11_ABI": "GLIBCXX_USE_CXX11_ABI",
        }
        additional_options.update(
            {
                # Build options that have the same environment variable name and CMake variable name and that do not start
                # with "BUILD_", "USE_", or "CMAKE_". If you are adding a new build option, also make sure you add it to
                # CMakeLists.txt.
                var: var
                for var in (
                    "BLAS",
                    "BUILDING_WITH_TORCH_LIBS",
                    "EXPERIMENTAL_SINGLE_THREAD_POOL",
                    "INSTALL_TEST",
                    "MKL_THREADING",
                    "MKLDNN_CPU_RUNTIME",
                    "Numa_INCLUDE_DIR",
                    "Numa_LIBRARIES",
                    "ATEN_THREADING",
                    "WERROR",
                    "OPENSSL_ROOT_DIR",
                    "STATIC_DISPATCH_BACKEND",
                )
            }
        )

        for var, val in my_env.items():
            # We currently pass over all environment variables that start with "BUILD_", "USE_", and "CMAKE_". This is
            # because we currently have no reliable way to get the list of all build options we have specified in
            # CMakeLists.txt. (`cmake -L` won't print dependent options when the dependency condition is not met.) We
            # will possibly change this in the future by parsing CMakeLists.txt ourselves (then additional_options would
            # also not be needed to be specified here).
            true_var = additional_options.get(var)
            if true_var is not None:
                build_options[true_var] = val
            elif var.startswith(
                ("BUILD_", "USE_", "CMAKE_", "GENERATED_", "ENABLE_")
            ) or var.endswith(("EXITCODE", "EXITCODE__TRYRUN_OUTPUT", "INSTALL_DIR")):
                build_options[var] = val

        # The default value cannot be easily obtained in CMakeLists.txt. We set it here.
        py_lib_path = sysconfig.get_path("purelib")
        cmake_prefix_path = build_options.get("CMAKE_PREFIX_PATH", None)
        if cmake_prefix_path:
            build_options["CMAKE_PREFIX_PATH"] = (
                py_lib_path + ";" + cast(str, cmake_prefix_path)
            )
        else:
            build_options["CMAKE_PREFIX_PATH"] = py_lib_path

        # Some options must be post-processed. Ideally, this list will be shrunk to only one or two options in the
        # future, as CMake can detect many of these libraries pretty comfortably. We have them here for now before CMake
        # integration is completed. They appear here not in the CMake.defines call below because they start with either
        # "BUILD_" or "USE_" and must be overwritten here.
        build_options.update(
            {
                # Note: Do not add new build options to this dict if it is directly read from environment variable -- you
                # only need to add one in `CMakeLists.txt`. All build options that start with "BUILD_", "USE_", or "CMAKE_"
                # are automatically passed to CMake; For other options you can add to additional_options above.
                "USE_PYTHON": build_python,
                "BUILD_TEST": build_test,
            }
        )

        # Options starting with CMAKE_
        cmake__options = {
            "CMAKE_INSTALL_PREFIX": install_dir,
        }

        # We set some CMAKE_* options in our Python build code instead of relying on the user's direct settings. Emit an
        # error if the user also attempts to set these CMAKE options directly.
        specified_cmake__options = set(build_options).intersection(cmake__options)
        if len(specified_cmake__options) > 0:
            print(
                ", ".join(specified_cmake__options)
                + " should not be specified in the environment variable. They are directly set by PyTorch build script."
            )
            sys.exit(1)
        build_options.update(cmake__options)

        CMake.defines(
            args,
            PYTHON_INCLUDE_DIR=sysconfig.get_path("include"),
            **build_options,
        )

        for env_var_name in my_env:
            if env_var_name.startswith("gh"):
                # github env vars use utf-8, on windows, non-ascii code may
                # cause problem, so encode first
                try:
                    my_env[env_var_name] = str(my_env[env_var_name].encode("utf-8"))
                except UnicodeDecodeError as e:
                    shex = ":".join(
                        "{:02x}".format(ord(c)) for c in my_env[env_var_name]
                    )
                    print(
                        "Invalid ENV[{}] = {}".format(env_var_name, shex),
                        file=sys.stderr,
                    )
                    print(e, file=sys.stderr)
        args.append(base_dir)
        self.run(args, env=my_env)
