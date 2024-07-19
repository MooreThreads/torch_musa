"""The musa extension file"""

import copy
import multiprocessing
import os
import shlex
import subprocess
import sys
import warnings
from os.path import dirname, realpath, join
from typing import List, Optional, Tuple

import setuptools
import torch
import torch._appdirs
from setuptools.command.build_ext import build_ext
from torch.torch_version import TorchVersion
from torch.utils.cpp_extension import (
    is_ninja_available,
    get_compiler_abi_compatibility_and_version,
    verify_ninja_availability,
    _run_ninja_build,
)
import torch_musa
from torch_musa.core._utils import _get_musa_arch

if os.getenv("MAX_JOBS") is None:
    os.environ["MAX_JOBS"] = str(multiprocessing.cpu_count())

SUBPROCESS_DECODE_ARGS = ()


def get_cxx_compiler():
    compiler = os.environ.get("CXX", "c++")
    return compiler


def _maybe_write(filename, new_content):
    r"""
    Equivalent to writing the content into the file but will not touch the file
    if it already had the right content (to avoid triggering recompile).
    """
    if os.path.exists(filename):
        with open(filename, encoding="utf-8") as f:
            content = f.read()

        if content == new_content:
            # The file already contains the right thing!
            return

    with open(filename, "w", encoding="utf-8") as source_file:
        source_file.write(new_content)


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    musa_cflags,
    musa_post_cflags,
    musa_dlink_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_musa,
    force_mcc,
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `musa_cflags`: list of flags to pass to $nvcc. Can be None.
    `musa_postflags`: list of flags to append to the $nvcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_musa`: If we should be compiling with MUSA.
    `force_mcc`: Force compile .cpp files with mcc compiler
    """

    def sanitize_flags(flags):
        if flags is None:
            return []
        return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    musa_cflags = sanitize_flags(musa_cflags)
    musa_post_cflags = sanitize_flags(musa_post_cflags)
    musa_dlink_post_cflags = sanitize_flags(musa_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ["ninja_required_version = 1.3", f"cxx = {compiler}"]
    if with_musa or musa_dlink_post_cflags:
        if "PYTORCH_MCC" in os.environ:
            mcc = os.getenv(
                "PYTORCH_MCC"
            )  # user can set mcc compiler with ccache using the environment variable here
        else:
            mcc = _join_musa_home("bin", "mcc")
        config.append(f"mcc = {mcc}")

    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_musa:
        flags.append(f'musa_cflags = {" ".join(musa_cflags)}')
        flags.append(f'musa_post_cflags = {" ".join(musa_post_cflags)}')
    flags.append(f'musa_dlink_post_cflags = {" ".join(musa_dlink_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ["rule compile"]

    if force_mcc:
        compile_rule.append(
            "  command = $mcc -x musa -MMD -MF $out.d $musa_cflags -c $in -o $out $musa_post_cflags"
        )
    else:
        compile_rule.append(
            "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags"
        )
    compile_rule.append("  depfile = $out.d")
    compile_rule.append("  deps = gcc")

    if with_musa:
        musa_compile_rule = ["rule musa_compile"]
        mcc_gendeps = ""
        # --generate-dependencies-with-compile is not supported by ROCm
        if torch.version.musa is not None:
            musa_compile_rule.append("  depfile = $out.d")
            musa_compile_rule.append("  deps = gcc")
            mcc_gendeps = "-MD -MF $out.d"
        musa_compile_rule.append(
            f"  command = $mcc {mcc_gendeps} $musa_cflags -c $in -o $out $musa_post_cflags"
        )

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_musa_source = _is_musa_file(source_file) and with_musa
        rule = "musa_compile" if is_musa_source else "compile"
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f"build {object_file}: {rule} {source_file}")

    if musa_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), "dlink.o")
        devlink_rule = ["rule musa_devlink"]
        devlink_rule.append("  command = $mcc $in -o $out $musa_dlink_post_cflags")
        devlink = [f'build {devlink_out}: musa_devlink {" ".join(objects)}']
        objects += [devlink_out]
    else:
        devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ["rule link"]
        link_rule.append("  command = $cxx $in $ldflags -o $out")

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f"default {library_target}"]
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_musa:
        blocks.append(musa_compile_rule)
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)


def _write_ninja_file_and_compile_objects(
    sources: List[str],
    objects,
    cflags,
    post_cflags,
    musa_cflags,
    musa_post_cflags,
    musa_dlink_post_cflags,
    build_directory: str,
    verbose: bool,
    with_musa: Optional[bool],
    force_mcc: Optional[bool],
) -> None:
    verify_ninja_availability()

    compiler = get_cxx_compiler()

    get_compiler_abi_compatibility_and_version(compiler)
    if with_musa is None:
        with_musa = any(map(_is_musa_file, sources))
    build_file_path = os.path.join(build_directory, "build.ninja")
    if verbose:
        print(f"Emitting ninja build file {build_file_path}...", file=sys.stderr)
    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        musa_cflags=musa_cflags,
        musa_post_cflags=musa_post_cflags,
        musa_dlink_post_cflags=musa_dlink_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_musa=with_musa,
        force_mcc=force_mcc,
    )
    if verbose:
        print("Compiling objects...", file=sys.stderr)
    _run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix="Error compiling objects for extension",
    )


# pylint: disable=W0718
def _find_musa_home() -> Optional[str]:
    """Find the MUSA install path."""
    # Guess #1
    musa_home = os.environ.get("MUSA_HOME") or os.environ.get("MUSA_PATH")
    if musa_home is None:
        # Guess #2
        try:
            which = "which"
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                mcc = (
                    subprocess.check_output([which, "mcc"], stderr=devnull)
                    .decode(*SUBPROCESS_DECODE_ARGS)
                    .rstrip("\r\n")
                )
                musa_home = os.path.dirname(os.path.dirname(mcc))
        except Exception:
            # Guess #3
            musa_home = "/usr/local/musa"
            if not os.path.exists(musa_home):
                musa_home = None
    if musa_home and not torch.musa.is_available():
        print(
            f"No MUSA runtime is found, using MUSA_HOME='{musa_home}'", file=sys.stderr
        )
    return musa_home


RERUN_CMAKE = False
install_requires = ["packaging"]
MUSA_HOME = _find_musa_home()
MUDNN_HOME = os.environ.get("MUDNN_HOME") or os.environ.get("MUDNN_PATH")


def include_paths(musa: bool = False) -> List[str]:
    """
    Get the include paths required to build a C++ or MUSA extension.

    Args:
        musa: If `True`, includes MUSA-specific include paths.

    Returns:
        A list of include path strings.
    """
    torch_musa_dir_path = realpath(dirname(dirname(dirname(__file__))))
    paths = [
        join(torch_musa_dir_path, path)
        for path in [
            "/usr/local/musa/include",
            "torch_musa/share/generated_cuda_compatible/aten/src",
            "torch_musa/share/generated_cuda_compatible/include",
            "torch_musa/share/generated_cuda_compatible/include/torch/csrc/api/include",
            "torch_musa/share/torch_musa_codegen",
        ]
    ] + [torch_musa_dir_path]

    if musa:
        musa_home_include = _join_musa_home("include")
        paths.append(musa_home_include)
        if MUDNN_HOME is not None:
            paths.append(os.path.join(MUDNN_HOME, "include"))
    return paths


def library_paths(musa: bool = False) -> List[str]:
    """
    Get the library paths required to build a C++ or MUSA extension.

    Args:
        musa: If `True`, includes MUSA-specific library paths.

    Returns:
        A list of library path strings.
    """
    # We need to link against libtorch.so
    paths = [
        join(dirname(torch.__file__), "lib"),
        join(dirname(torch_musa.__file__), "lib"),
    ]
    lib_dir = "lib64"
    if musa:
        if not os.path.exists(_join_musa_home(lib_dir)) and os.path.exists(
            _join_musa_home("lib")
        ):
            lib_dir = "lib"
            paths.append(_join_musa_home(lib_dir))
            if MUDNN_HOME is not None:
                paths.append(os.path.join(MUDNN_HOME, lib_dir))
    return paths


# pylint: disable=C0103, C0116
def MUSAExtension(name, sources, *args, **kwargs):
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += library_paths(musa=True)
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    libraries.append("torch")
    libraries.append("torch_cpu")
    libraries.append("torch_python")
    libraries.append("musa_python")
    kwargs["libraries"] = libraries

    include_dirs = kwargs.get("include_dirs", [])

    include_dirs += include_paths(musa=True)
    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    dlink_libraries = kwargs.get("dlink_libraries", [])
    dlink = kwargs.get("dlink", False) or dlink_libraries
    if dlink:
        extra_compile_args = kwargs.get("extra_compile_args", {})

        extra_compile_args_dlink = extra_compile_args.get("mcc_dlink", [])
        extra_compile_args_dlink += ["-dlink"]
        extra_compile_args_dlink += [f"-L{x}" for x in library_dirs]
        extra_compile_args_dlink += [f"-l{x}" for x in dlink_libraries]
        extra_compile_args["mcc_dlink"] = extra_compile_args_dlink

        kwargs["extra_compile_args"] = extra_compile_args

    # the following is equivalent to -R
    extra_link_args = kwargs.get("extra_link_args", [])
    torch_lib_path = join(dirname(torch.__file__), "lib")
    torch_musa_lib_path = join(dirname(torch_musa.__file__), "lib")

    extra_link_args = (
        extra_link_args
        + ["-Wl,-rpath,$ORIGIN/lib"]
        + [f"-Wl,-rpath,{torch_lib_path}"]
        + [f"-Wl,-rpath,{torch_musa_lib_path}"]
    )

    kwargs["extra_link_args"] = extra_link_args

    user_define_macros = kwargs.get("define_macros", [])
    user_define_macros += [("TORCH_MUSA_ARCH", str(_get_musa_arch() * 10))]
    kwargs["define_macros"] = user_define_macros

    if "extra_compile_args" in kwargs:
        mcc_user_extra_compile_args = kwargs["extra_compile_args"].get("mcc", [])
        mcc_user_extra_compile_args += ["--cuda-gpu-arch=mp_" + str(_get_musa_arch())]
        kwargs["extra_compile_args"]["mcc"] = mcc_user_extra_compile_args
    else:
        kwargs["extra_compile_args"] = {}
        kwargs["extra_compile_args"]["mcc"] = [
            "--cuda-gpu-arch=mp_" + str(_get_musa_arch())
        ]
    if "cxx" not in kwargs["extra_compile_args"]:
        kwargs["extra_compile_args"]["cxx"] = []
    kwargs["extra_compile_args"]["mcc"].append("-march=native")
    kwargs["extra_compile_args"]["cxx"].append("-march=native")
    return setuptools.Extension(name, sources, *args, **kwargs)


COMMON_MCC_FLAGS = [
    # It is copied from cuda, but musa now support half and bf16, so we comment it.
    # '-D__MUSA_NO_HALF_OPERATORS__',
    # '-D__MUSA_NO_HALF_CONVERSIONS__',
    # '-D__MUSA_NO_BFLOAT16_CONVERSIONS__',
    # '-D__MUSA_NO_HALF2_OPERATORS__'
]


def _is_musa_file(path: str) -> bool:
    valid_ext = [".mu", ".muh"]
    return os.path.splitext(path)[1] in valid_ext


def _join_musa_home(*paths) -> str:
    """
    Join paths with MUSA_HOME, or raises an error if it MUSA_HOME is not set.

    This is basically a lazy way of raising an error for missing $MUSA_HOME
    only once we need to get any MUSA-specific path.
    """
    if MUSA_HOME is None:
        raise OSError(
            "MUSA_HOME environment variable is not set. "
            "Please set it to your MUSA install root."
        )
    return os.path.join(MUSA_HOME, *paths)


# pylint: disable = C0200, E0601, W0612, W0612, W0237
class BuildExtension(build_ext):
    """
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++17``) as well as mixed
    C++/MUSA compilation (and support for MUSA files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``mcc``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and MUSA compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    """

    @classmethod
    def with_options(cls, **options):
        """
        Return a subclass with alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        """

        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get("use_ninja", True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = (
                "Attempted to use ninja as the BuildExtension backend but "
                "{}. Falling back to using the slow distutils backend."
            )
            if not is_ninja_available():
                warnings.warn(msg.format("we could not find ninja."))
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:
        compiler_name, compiler_version = self._check_abi()

        musa_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not musa_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == ".mu":
                    musa_ext = True
                    break
            extension = next(extension_iter, None)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' and 'mcc' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx' and 'mcc' is
            # passed to extra_compile_args in MUSAExtension, i.e.
            #   MUSAExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   MUSAExtension(..., extra_compile_args={'mcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ["cxx", "mcc"]:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, "-DTORCH_API_INCLUDE_EXTENSION_H")
            # See note [Pybind11 ABI constants]
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}")
                if val is not None:
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

            if "mcc_dlink" in extension.extra_compile_args:
                assert (
                    self.use_ninja
                ), f"With dlink=True, ninja is required to build musa extension {extension.name}."

        # Register .mu, .muh as valid source extensions.
        self.compiler.src_extensions += [".mu", ".muh"]
        # Save the original _compile method for later.
        original_compile = self.compiler._compile

        def append_std17_if_no_std_present(cflags) -> None:
            # MCC does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = "-{}="
            cpp_flag_prefix = cpp_format_prefix.format("std")
            cpp_flag = cpp_flag_prefix + "c++17"
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_musa_flags(cflags):
            # mcc doesn't support -ccbin then we remove it
            return COMMON_MCC_FLAGS + ["-fPIC"] + cflags

        def convert_to_absolute_paths_inplace(paths):
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(
            obj, src, ext, cc_args, extra_postargs, pp_opts
        ) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_musa_file(src):
                    mcc = [_join_musa_home("bin", "mcc")]
                    self.compiler.set_executable("compiler_so", mcc)
                    if isinstance(cflags, dict):
                        cflags = cflags["mcc"]
                    cflags = unix_musa_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]
                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable("compiler_so", original_compiler)

        def unix_wrap_ninja_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            r"""Compiles sources by outputting a ninja file and running it."""
            # NB: I copied some lines from self.compiler (which is an instance
            # of distutils.UnixCCompiler). See the following link.
            # https://github.com/python/cpython/blob/f03a8f8d5001963ad5b5b28dbd95497e9cc15596/Lib/distutils/ccompiler.py#L564-L567
            # This can be fragile, but a lot of other repos also do this
            # (see https://github.com/search?q=_setup_compile&type=Code)
            # so it is probably OK; we'll also get CI signal if/when
            # we update our python version (which is when distutils can be
            # upgraded)

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = self.compiler._setup_compile(
                output_dir, macros, include_dirs, sources, depends, extra_postargs
            )
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_musa = any(map(_is_musa_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/mcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs["cxx"]
            else:
                post_cflags = list(extra_postargs)
            append_std17_if_no_std_present(post_cflags)
            force_mcc = False
            if "force_mcc" in post_cflags:
                force_mcc = True
                with_musa = True
                post_cflags.remove("force_mcc")

            musa_post_cflags = None
            musa_cflags = None
            if with_musa:
                musa_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    musa_post_cflags = extra_postargs["mcc"]
                else:
                    musa_post_cflags = list(extra_postargs)

                musa_post_cflags = unix_musa_flags(musa_post_cflags)
                append_std17_if_no_std_present(musa_post_cflags)
                musa_cflags = [shlex.quote(f) for f in musa_cflags]
                musa_post_cflags = [shlex.quote(f) for f in musa_post_cflags]

            if isinstance(extra_postargs, dict) and "mcc_dlink" in extra_postargs:
                musa_dlink_post_cflags = unix_musa_flags(extra_postargs["mcc_dlink"])
            else:
                musa_dlink_post_cflags = None
            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                musa_cflags=musa_cflags,
                musa_post_cflags=musa_post_cflags,
                musa_dlink_post_cflags=musa_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_musa=with_musa,
                force_mcc=force_mcc,
            )

            # Return *all* object filenames, not just the ones we just built.
            return objects

        if self.use_ninja:
            self.compiler.compile = unix_wrap_ninja_compile
        else:
            self.compiler._compile = unix_wrap_single_compile
        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super().get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split(".")
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = ".".join(without_abi)
        return ext_filename

    def _check_abi(self) -> Tuple[str, TorchVersion]:
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, "compiler_cxx"):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = get_cxx_compiler()
        _, version = get_compiler_abi_compatibility_and_version(compiler)
        return compiler, version

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_torch_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split(".")
        name = names[-1]
        define = f"-DTORCH_EXTENSION_NAME={name}"
        self._add_compile_flag(extension, define)

    def _add_gnu_cpp_abi_flag(self, extension):
        # use the same CXX ABI as what PyTorch was compiled with
        self._add_compile_flag(
            extension,
            "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        )
