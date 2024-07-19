## Code Formatting for Torch Musa

### CMake formatting

[cmakelang](https://cmake-format.readthedocs.io/en/latest/) is used to format CMake files.

You can use the following command to format CMake files.
```bash
cd torch_musa
pip install -r requirements.txt
cmake-format -c ./.cmake-format.yaml -i ./CMakeLists.txt  ./torch_musa/csrc/CMakeLists.txt

or 

pre-commit run cmake-format --all-files
```

### Python Formatting

`black` is used to format python files. Section `[tool.black]` in `project.toml` is used to configure the style.
To format the python file:

```bash
pre-commit run black --all-files
```

### Python Lint

`pylint` is used to lint the python files. `tools/lint/pylintrc` is used as the configuration file.

### Cpp Formatting

TBD.
