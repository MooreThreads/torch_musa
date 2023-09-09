#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
scripts_dir="$(dirname "$(realpath "${BASH_SOURCE:-$0}" )")"
torch_musa_dir="$(dirname "$scripts_dir")"

####### musa toolkits install, this needs latest stable DDK.
musa_install_dir=/usr/local/musa
[ -d "${musa_install_dir}" ] && rm -rf "${musa_install_dir}"
musa_toolkits_path="./daily_musa_toolkits_${DATE}"
musa_toolkits_oss_link="https://oss.mthreads.com/release-ci/computeQA/musa/newest/musa_toolkits_install_full.tar.gz"

wget --no-check-certificate ${musa_toolkits_oss_link} -P ${musa_toolkits_path}
tar -xvzf ${musa_toolkits_path}/musa_toolkits_install_full.tar.gz -C ${musa_toolkits_path}

pushd ${musa_toolkits_path}/musa_toolkits_install
bash install.sh
popd

####### end

####### install math components
bash ${torch_musa_dir}/docker/common/install_math.sh
####### end

####### install mccl
bash ${torch_musa_dir}/docker/common/install_mccl.sh
####### end


####### update the muAlg
mualg_path="./daily_mualg_${DATE}"
mualg_oss_link="https://oss.mthreads.com/release-ci/computeQA/mathX/newest/mualg.tar"
wget --no-check-certificate ${mualg_oss_link} -P ${mualg_path}
tar -xvf ${mualg_path}/mualg.tar -C ${mualg_path}

pushd ${mualg_path}/package
# running in root permission
for DEB_NAME in *.deb; do
  dpkg -i ${DEB_NAME}
done
popd

####### end

####### update mcc to latest
musa_newest_path="http://oss.mthreads.com/release-ci/computeQA/musa/newest"
mtcc_pkg_release="mtcc-nightly-x86_64-linux-gnu-ubuntu-20.04.tar.gz"
musa_path=${musa_newest_path}
mtcc_pkg=${mtcc_pkg_release}
download_dir=${scripts_dir}/download/
mtcc_dir="mtcc"
mkdir -p "${download_dir}/${mtcc_dir}"
pushd "${download_dir}/${mtcc_dir}"
wget --no-check-certificate "${musa_path}/${mtcc_pkg}" -P ./
tar -xvf "${mtcc_pkg}"
./install.sh
popd

####### end
