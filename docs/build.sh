#!/bin/bash

script_path=`dirname $0`
pushd $script_path
pushd developer_guide
rm -rf build
./makelatexpdf.sh
cp build/latex/*.pdf ../
popd
popd
