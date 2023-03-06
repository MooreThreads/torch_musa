import os
import json
import subprocess

from setuptools import setup, find_packages

json_file = os.path.join(cwd, "scripts/configs", "build.conf")
torch_musa_version = "unknown"
if os.path.isfile(json_file):
    with open(json_file, "r") as f:  # pylint: disable=W1514
        json_dict = json.load(f)
        torch_musa_version = json_dict["version"].strip()

# Setup
setup(
    name="torch_musa",
    version=torch_musa_version,
    description="A PyTorch backend extension for Moore Threads MUSA",
    packages=["torch_musa"],
    ext_modules=[],
    package_data={},
    extras_require={},
    cmdclass={"build_ext": Build, "clean": Clean, "install": Install},
)
