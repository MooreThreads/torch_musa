#!/bin/bash
set -e

CUR_DIR=$(realpath "$(dirname $0)/..")
TORCH_MUSA_HOME=$CUR_DIR
PYTORCH_PATH=${PYTORCH_REPO_PATH:-${TORCH_MUSA_HOME}/../pytorch}
PATCHES_DIR=${TORCH_MUSA_HOME}/torch_patches

if [ ! -d $PYTORCH_PATH ]
then
  echo "PYTORCH_PATH ($PYTORCH_PATH) does not exists"
  echo "Set PYTORCH_PATH by: export PYTORCH_PATH=/path/to/pytorch" 
  exit 1
fi

if [ ! -d $PATCHES_DIR ]
then
  echo "PATCHES_DIR ($PATCHES_DIR) does not exists"
  echo "Set PATCHES_DIR with TORCH_MUSA_HOME: export TORCH_MUSA_HOME=/path/to/torch_musa" 
  exit 1
fi

pushd $PYTORCH_PATH
diff_files=$(git diff --name-only)
for diff_file in ${diff_files[@]}
do
  patch_file_name="${PATCHES_DIR}/$(echo $diff_file | rev | cut -f1 -d'/' | rev).patch"
  if [ -f $patch_file_name ]
  then
    echo "Updating patch ${patch_file_name}"
  else
    echo "Generating patch ${patch_file_name}"
  fi
  git diff $diff_file > ${patch_file_name}
done

echo "pytorch patches generated"
exit 0
