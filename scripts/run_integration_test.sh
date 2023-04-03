#!/bin/bash --login
set -exo pipefail

# Integration tests
pytest -s tests/integration/vision
