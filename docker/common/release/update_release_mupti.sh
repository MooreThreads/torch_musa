#!/bin/bash
set -e

DATE=$(date +%Y%m%d)
mupti_path=./release_mupti_${DATE}

# TODO(the oss link will be updated later)
mupti_oss_link=${mupti_oss_link:-https://oss.mthreads.com/release-ci/computeQA/musa/MUPTI/MUPTI_torch_profiler.tar.gz}
if [ "${KUAE}" ]; then
    echo -e "\033[31mIn kuae env, then use mupti for kuae! \033[0m"
    mupti_oss_link=https://oss.mthreads.com/release-ci/computeQA/musa/MUPTI/MUPTI_torch_profiler_kuae.tar.gz
fi

wget --no-check-certificate ${mupti_oss_link} -P ${mupti_path}
tar -xvzf ${mupti_path}/MUPTI_*.tar.gz -C ${mupti_path}
pushd ${mupti_path}
dpkg -i MUPTI-0.2.0-Linux.deb
popd
echo -e "\033[31mmupti update to the newest version! \033[0m"
rm -rf ${mupti_path}
rm -f MUPTI-0.2.0-Linux.deb
