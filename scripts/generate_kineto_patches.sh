#!/bin/bash
set -e

CUR_DIR=$(realpath "$(dirname $0)/..")
TORCH_MUSA_HOME=$CUR_DIR
KINETO_PATH=${TORCH_MUSA_HOME}/../pytorch/third_party/kineto
PATCHES_DIR=${TORCH_MUSA_HOME}/kineto_patches

if [ ! -d $PATCHES_DIR ]
then
  echo "PATCHES_DIR ($PATCHES_DIR) does not exists"
  echo "Set PATCHES_DIR with TORCH_MUSA_HOME: export TORCH_MUSA_HOME=/path/to/torch_musa" 
  exit 1
fi

pushd $KINETO_PATH
git add .  # New files support
diff_files=$(git diff HEAD --name-only)
for diff_file in ${diff_files[@]}
do
  patch_file_name="${PATCHES_DIR}/$(echo $diff_file | sed 's/\//_/g' ).patch"
  if [ -f $patch_file_name ]
  then
    echo "Updating patch ${patch_file_name}"
  else
    echo "Generating patch ${patch_file_name}"
  fi
  git diff --cached -- $diff_file > ${patch_file_name}
done
git reset HEAD # Ready for next apply-patches

echo "kineto patches generated"
exit 0
