#!/bin/bash --login

set -exo pipefail

# Install torch musa
python3 setup.py install

# Unit tests
python3 -m unittest discover tests/unittest "test_*.py"
