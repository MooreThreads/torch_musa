#!/bin/bash --login
set -x

LOCAL_DATA_ROOT="/data/torch_musa_integration/local"
SHARED_DATA_ROOT="/data/torch_musa_integration/shared"
SELECTED_DATA_ROOT=""

if [ -d "${LOCAL_DATA_ROOT}" ] && [ -n "$(ls -A "${LOCAL_DATA_ROOT}")" ]; then
  SELECTED_DATA_ROOT="${LOCAL_DATA_ROOT}"
  echo -e "Found local integration data root: ${SELECTED_DATA_ROOT}"
elif [ -d "${SHARED_DATA_ROOT}" ] && [ -n "$(ls -A "${SHARED_DATA_ROOT}")" ]; then
  SELECTED_DATA_ROOT="${SHARED_DATA_ROOT}"
  echo -e "Found shared integration data root: ${SELECTED_DATA_ROOT}"
else
  echo -e "Not found local/shared integration data"
  exit 1
fi

export INTEGRATION_DATA_ROOT="${SELECTED_DATA_ROOT}"

TEST_REPORT_DIR=build/reports/integration_test
mkdir -p ${TEST_REPORT_DIR}
GPU_TYPE=${GPU_TYPE:-S3000}
# Integration tests
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-resnet-test-failed-results.xml -sv tests/integration/vision/resnet
pytest --last-failed --junitxml=${TEST_REPORT_DIR}/${GPU_TYPE}-bert-test-failed-results.xml -sv tests/integration/nlp/bert
