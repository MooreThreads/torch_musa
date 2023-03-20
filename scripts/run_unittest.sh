#!/bin/bash --login

set -exo pipefail

# Install torch musa
MAX_JOBS=1 bash ./build.sh

# Unit tests

python3 -m unittest discover tests/unittest "test_*.py"
