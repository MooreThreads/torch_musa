#!/bin/bash --login
set -exo pipefail

# Unit tests
pytest -s tests/unittest/operator
