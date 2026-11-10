"""AOTI standalone execution plus torch_musa/inductor integration checks."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
import torch_musa


class TestAOTIStandalone(unittest.TestCase):
    def _run(self, command, **kwargs):
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **kwargs,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def _configure_native(self, build):
        torch_root = Path(torch.__file__).resolve().parent
        musa_root = Path(torch_musa.__file__).resolve().parent
        self._run(
            [
                "cmake",
                "-G",
                "Ninja",
                "-S",
                str(Path(__file__).parent / "aoti_standalone"),
                "-B",
                str(build),
                f"-DTorch_DIR={torch_root / 'share/cmake/Torch'}",
                f"-DTorchMusa_DIR={musa_root / 'share/cmake/TorchMusa'}",
                "-DCMAKE_DISABLE_FIND_PACKAGE_Python=TRUE",
                "-DCMAKE_DISABLE_FIND_PACKAGE_Python3=TRUE",
            ]
        )

    @staticmethod
    def _native_environment():
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"):
            env.pop(key, None)
        env["PATH"] = "/nonexistent"  # No interpreter or subprocess tools.
        env["LD_BIND_NOW"] = "1"
        env["LD_LIBRARY_PATH"] = ":".join(
            [
                str(Path(torch_musa.__file__).resolve().parent / "lib"),
                str(Path(torch.__file__).resolve().parent / "lib"),
                str(Path(os.environ.get("MUSA_HOME", "/usr/local/musa")) / "lib"),
            ]
        )
        return env

    @unittest.skipUnless(sys.platform == "linux", "Uses Linux ELF and /proc checks")
    def test_backend_registration_without_python(self):
        for tool in ("cmake", "ninja"):
            if shutil.which(tool) is None:
                self.skipTest(f"{tool} is required")
        with tempfile.TemporaryDirectory(prefix="musa_backend_registration_") as temp:
            build = Path(temp) / "build"
            self._configure_native(build)
            self._run(
                ["cmake", "--build", str(build), "--target", "backend_registration"]
            )
            output = self._run(
                [str(build / "backend_registration")],
                env=self._native_environment(),
                timeout=120,
            )
            self.assertIn("PASS: standalone MUSA backend registered", output)

    def test_inductor_python_binding(self):
        from torch._inductor import config
        from torch._inductor.codecache import CppPythonBindingsCodeCache
        from torch._inductor.cpp_builder import get_cpp_compiler

        if shutil.which(get_cpp_compiler()) is None:
            self.skipTest("A C++ compiler is required")

        # Importing torch_musa must preserve the upstream Python include paths
        # used by ordinary JIT kernels, including CPU kernels.
        with config.patch(cpp_cache_precompile_headers=False):
            kernel = CppPythonBindingsCodeCache.load_pybinding(
                [], 'extern "C" void kernel() {}', needs_vec_isa=False
            )
            kernel()

    @unittest.skipUnless(sys.platform == "linux", "Uses Linux ELF and /proc checks")
    def test_export_and_run_without_python(self):
        if not torch.musa.is_available():
            self.skipTest("MUSA device is required")
        for tool in ("cmake", "ninja", "readelf", "nm"):
            if shutil.which(tool) is None:
                self.skipTest(f"{tool} is required")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "weight", torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
                )

            def forward(self, x):
                return torch.relu(torch.mm(x, self.weight) - 0.5)

        run = self._run

        def check_elf(path):
            dynamic = run(["readelf", "-d", str(path)])
            needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
            self.assertFalse(
                any("python" in library.lower() for library in needed), needed
            )
            symbols = run(["nm", "-D", "--undefined-only", str(path)])
            self.assertIsNone(re.search(r"\b(?:_?Py\w*|THP\w*)\b", symbols), symbols)
            return needed

        with tempfile.TemporaryDirectory(prefix="musa_aoti_standalone_") as temp:
            root = Path(temp)
            model = Model().eval().to("musa")
            x = torch.arange(16, dtype=torch.float32, device="musa").reshape(4, 4) / 16
            exported = torch.export.export(model, (x,))
            package = root / "model.pt2"
            torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(package),
                inductor_configs={
                    "aot_inductor.package_cpp_only": False,
                    "aot_inductor.dynamic_linkage": True,
                },
            )
            with zipfile.ZipFile(package) as archive:
                forbidden_package_references = (
                    b"libmusa_python",
                    b"musa_python",
                    b"libtorch_python",
                    b"torch_python",
                    b"libpython",
                )
                for member in archive.infolist():
                    payload = archive.read(member)
                    references = [
                        token
                        for token in forbidden_package_references
                        if token in payload
                    ]
                    self.assertFalse(
                        references,
                        f"Python dependency reference in {member.filename}: {references}",
                    )
                wrappers = [n for n in archive.namelist() if n.endswith(".wrapper.so")]
                self.assertTrue(wrappers, archive.namelist())
                for index, name in enumerate(wrappers):
                    elf = root / f"wrapper_{index}.so"
                    with archive.open(name) as source, elf.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    check_elf(elf)

            build = root / "build"
            self._configure_native(build)
            run(["cmake", "--build", str(build), "--parallel", "2"])
            executable = build / "aoti_standalone"
            self.assertTrue(
                any("libtorch_musa" in lib for lib in check_elf(executable)),
                "The CMake target must retain the MUSA backend",
            )
            commands = run(["ninja", "-C", str(build), "-t", "commands"])
            self.assertNotRegex(
                commands, r"(?:-I\s*|-isystem\s+)\S*/include/python\d"
            )
            self.assertNotRegex(commands, r"-l(?:musa_python|torch_python|python\d)")

            output = run(
                [str(executable), str(package)],
                env=self._native_environment(),
                timeout=120,
            )
            self.assertIn("PASS: standalone MUSA AOTI matches CPU", output)


if __name__ == "__main__":
    unittest.main()
