#!/bin/bash
# create softlink for *mkl.so files, because softlinks are missing in some mkl versions
# which invalids `USE_MKL=1` when compiling pytorch
directory=$1

mkl_so_files=$(find ${directory} -name *mkl*.so)
if [[ -n ${mkl_files} ]]; then
  exit 0
fi

mkl_so_1_files=$(find ${directory} -name *mkl*.so.1)
if [[ -n ${mkl_so_1_files} ]]; then
  suffix=".1"
  for file in $mkl_so_1_files
  do
    filename=$(basename ${file} $suffix)
    ln -s ${file} ${directory}/${filename}
  done
  exit 0
fi

mkl_so_2_files=$(find ${directory} -name *mkl*.so.2)
if [[ -n ${mkl_so_2_files} ]]; then
  suffix=".2"
  for file in $mkl_so_2_files
  do
    filename=$(basename ${file} $suffix)
    ln -s ${file} ${directory}/${filename}
  done
  exit 0
fi
