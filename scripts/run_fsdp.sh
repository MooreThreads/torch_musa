#!/bin/bash --login
set -exo pipefail
TEST_REPORT_DIR=build/reports/unit_test
mkdir -p ${TEST_REPORT_DIR}
GPU_TYPE=${GPU_TYPE:-S3000}
# Unit tests
start_time=$(date +%s.%N)
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-functorch-test-failed-results.xml -v tests/unittest/functorch
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-distributed-test-failed-results.xml -v tests/unittest/distributed
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-dynamo-test-failed-results.xml -v tests/unittest/dynamo
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-inductor-test-failed-results.xml -v tests/unittest/inductor
end_time=$(date +%s.%N)
runtime=$(( end_time - start_time ))
echo "Total runtime: $runtime seconds"
