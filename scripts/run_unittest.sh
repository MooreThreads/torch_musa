#!/bin/bash --login

set -exo pipefail

# Install perfmax
python setup.py install

# Unit tests
python -m unittest discover tests/unittest "test_*.py"
