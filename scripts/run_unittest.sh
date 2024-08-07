#!/bin/bash --login
set -exo pipefail
TEST_REPORT_DIR=build/reports/unit_test
mkdir -p ${TEST_REPORT_DIR}
GPU_TYPE=${GPU_TYPE:-S3000}
# Unit tests
start_time=$(date +%s.%N)
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-operator-test-failed-results.xml -v tests/unittest/operator
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-core-test-failed-results.xml -v tests/unittest/core
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-distributed-test-failed-results.xml -v tests/unittest/distributed
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-amp-test-failed-results.xml -v tests/unittest/amp
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-quantized-test-failed-results.xml -v tests/unittest/quantized
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-optim-test-failed-results.xml -v tests/unittest/optim
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-miscs-test-failed-results.xml -v tests/unittest/miscs
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-profiler-test-failed-results.xml -v tests/unittest/profiler/*.py
end_time=$(date +%s.%N)
runtime=$(echo "$end_time - $start_time" | bc)
echo "Total runtime: $runtime seconds"