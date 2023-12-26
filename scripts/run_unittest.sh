#!/bin/bash --login
set -exo pipefail

# Unit tests
pytest -v tests/unittest/operator
pytest -v tests/unittest/core
pytest -v tests/unittest/distributed
pytest -v tests/unittest/amp
pytest -v tests/unittest/quantized
pytest -v tests/unittest/optim
