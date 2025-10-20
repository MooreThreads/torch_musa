#!/bin/bash --login
set -exo pipefail
TEST_REPORT_DIR=build/reports/unit_test
mkdir -p ${TEST_REPORT_DIR}
GPU_TYPE=${GPU_TYPE:-S3000}
# Unit tests
start_time=$(date +%s.%N)
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-core-test-failed-results.xml -v tests/unittest/core
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-optim-test-failed-results.xml -v tests/unittest/optim
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-miscs-test-failed-results.xml -v tests/unittest/miscs
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-sparse-test-failed-results.xml -v tests/unittest/sparse
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-amp-test-failed-results.xml -v tests/unittest/amp
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-compile-test-failed-results.xml -v tests/unittest/compile
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-graph-test-failed-results.xml -v tests/unittest/graph
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-jit-test-failed-results.xml -v tests/unittest/jit
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-quantized-test-failed-results.xml -v tests/unittest/quantized

pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-profiler-test-failed-results.xml -v tests/unittest/profiler/*.py

find tests/unittest/standalone -name "*.py" | while read -r case;
do
  pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-standalone-test-failed-results.xml -v ${case}
done

end_time=$(date +%s.%N)
runtime=$(( end_time - start_time ))
echo "Total runtime: $runtime seconds"
