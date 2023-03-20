#!/bin/bash --login
set -exo pipefail

# Unit tests
python3 -m unittest discover tests/unittests "test_*.py"
