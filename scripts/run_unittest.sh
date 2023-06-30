#!/bin/bash --login
set -exo pipefail

# Unit tests
pytest -s tests/unittest/operator
pytest -s tests/unittest/core
pytest -s tests/unittest/distributed

