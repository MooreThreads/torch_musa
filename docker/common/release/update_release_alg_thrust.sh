#!/bin/bash
set -e

# clone mualg and install
read MUALG MUALG_TAG <<< `sed -n 5p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ $MUALG != "MUALG" ]; then
  echo -e "\033[31mload wrong muthrust version: $MUALG:$MUALG_TAG, check ./version.txt! \033[0m"
  exit 1
fi
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
read MUTHRUST MUTHRUST_TAG <<< `sed -n 6p $(dirname $0)/version.txt | awk -F: '{print $1, $2}'`
if [ $MUTHRUST != "MUTHRUST" ]; then
  echo -e "\033[31mload wrong muthrust version: $MUTHRUST:$MUTHRUST_TAG, check ./version.txt! \033[0m"
  exit 1
fi
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
