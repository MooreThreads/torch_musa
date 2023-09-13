#!/bin/bash --login
set -exo pipefail

# Unit tests
pytest -s tests/unittest/operator
pytest -s tests/unittest/core
pytest -s tests/unittest/distributed
pytest -s tests/unittest/amp
pytest -s tests/unittest/quantized
