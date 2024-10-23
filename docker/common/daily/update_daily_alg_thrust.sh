#!/bin/bash
set -e

# clone mualg and install
git clone -b musa-1.12.1 https://github.com/MooreThreads/muAlg --depth 1
pushd muAlg
./mt_build.sh -i
popd
echo -e "\033[31mualg update to the newest version! \033[0m"
rm -rf muAlg

# clone muthrust and install
git clone -b musa-1.12.1 https://github.com/MooreThreads/muThrust --depth 1
pushd muThrust
./mt_build.sh -i
popd
echo -e "\033[31muthrust update to the newest version! \033[0m"
rm -rf ./muThrust
