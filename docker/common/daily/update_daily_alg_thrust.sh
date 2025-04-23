#!/bin/bash
set -e

MUALG_TAG=musa-1.12.1
MUTHRUST_TAG=musa-1.12.1

# clone mualg and install
if [ -d /home/muAlg ]; then
  pushd /home/muAlg
  git checkout $MUALG_TAG
  ./mt_build.sh -i
  popd
else
  git clone -b $MUALG_TAG https://github.com/MooreThreads/muAlg --depth 1
  pushd muAlg
  ./mt_build.sh -i
  popd
fi
echo -e "\033[31mualg update to the newest version! \033[0m"
rm -rf muAlg

# clone muthrust and install
if [ -d /home/muThrust ]; then
  pushd /home/muThrust
  git checkout $MUTHRUST_TAG
  ./mt_build.sh -i
  popd
else
  git clone -b $MUTHRUST_TAG https://github.com/MooreThreads/muThrust --depth 1
  pushd muThrust
  ./mt_build.sh -i
  popd
fi
echo -e "\033[31muthrust update to the newest version! \033[0m"
rm -rf ./muThrust
