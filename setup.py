import os
import json
import subprocess

from distutils.command.clean import clean as clean_ext
from setuptools.command.install import install as install_ext
from setuptools.command.build_ext import build_ext


from setuptools import setup, find_packages

# Get the current working directory.
cwd = os.path.dirname(os.path.abspath(__file__))
json_file = os.path.join(cwd, "scripts/configs", "build.conf")
torch_musa_version = "unknown"
if os.path.isfile(json_file):
    with open(json_file, "r") as f:
        json_dict = json.load(f)
        torch_musa_version = json_dict["version"].strip()


class Build(build_ext):
    # TODO(mt-ai) Implement the build class.
    pass


class Clean(clean_ext):
    # TODO(mt-ai) Implement the clean class.
    pass


class Install(install_ext):
    # TODO(mt-ai) Implement the install class.
    pass


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
