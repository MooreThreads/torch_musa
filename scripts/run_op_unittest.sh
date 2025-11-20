#!/bin/bash --login
set -exo pipefail
export TORCH_MUSA_TESTING_NO_TF32=1
TEST_REPORT_DIR=build/reports/unit_test
mkdir -p ${TEST_REPORT_DIR}
GPU_TYPE=${GPU_TYPE:-S3000}
# Unit tests
start_time=$(date +%s.%N)
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-operator-test-failed-results.xml -v tests/unittest/operator

end_time=$(date +%s.%N)
runtime=$(( end_time - start_time ))
echo "Total runtime: $runtime seconds"
