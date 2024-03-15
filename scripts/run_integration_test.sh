#!/bin/bash --login
set -exo pipefail

# Integration tests
pytest -v tests/integration/vision
