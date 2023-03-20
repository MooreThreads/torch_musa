#!/bin/bash --login
set -exo pipefail

# Unit tests
python3 -m unittest discover tests/unittest "test_*.py"
