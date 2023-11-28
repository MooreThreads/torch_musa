#!/bin/bash
set -e
# set -x

current="$(dirname "$(realpath "${BASH_SOURCE:-$0}" )")"
download_dir=${current}/download/
musa_install_dir="/usr/local/musa"
echo "----------------------------------------------------------------"
echo "current  dir  : ${current}"
echo "download dir  : ${download_dir}"
echo "install  dir  : ${musa_install_dir}"
echo "----------------------------------------------------------------"

do_install_kmd="false"
do_install_umd="false"
do_install_murt="false"
do_install_mtcc="false"
do_download_pkg="true"

# include umd kmd vps
ddk_history_path="http://oss.mthreads.com/release-ci/computeQA/ddk/history"
ddk_stable_path="http://oss.mthreads.com/release-ci/computeQA/ddk/stable"
ddk_newest_path="http://oss.mthreads.com/release-ci/computeQA/ddk/newest"

# include mtcc / runtime / musa_toolkits
musa_history_path="http://oss.mthreads.com/release-ci/computeQA/musa/history"
# musa_stable_path="http://oss.mthreads.com/release-ci/computeQA/musa/stable"
musa_newest_path="http://oss.mthreads.com/release-ci/computeQA/musa/newest"

# deprecated
# daily_path="http://oss.mthreads.com/release-ci/test-compute/daily_pkg"

# mtcc-nightly-x86_64-linux-gnu-ubuntu-20.04.tar.gz
# MUSA-Runtime_use_armory.tar.gz
# MUSA-Runtime_debug_pdump.tar.gz
# kmd-vps_x86-mtgpu_linux-xorg-release-pdump_off.tar
# kmd-sudi_x86-mtgpu_linux-xorg-release-pdump_off.tar
# kmd-sudi_x86-mtgpu_linux-xorg-debug-pdump_on.tar
# kmd-sudi_x86-mtgpu_linux-xorg-debug-pdump_off.tar
# umd_x86-mtgpu_linux-xorg-debug-pdump_off.tar
# umd_x86-mtgpu_linux-xorg-debug-pdump_on.tar
# umd_x86-mtgpu_linux-xorg-release-pdump_off_gcov.tar
# umd_x86-mtgpu_linux-xorg-release-pdump_off.tar
# pdump_flag="release-pdump_off"
# murt_pkg="MUSA-Runtime_use_armory.tar.gz"

tool_pkg_release="20231125musa_toolkit6b27b102ccompute_musa_pkg7543/musa_toolkits_install.tar.gz"
mtcc_pkg_release="20231125mtccce3915698compute_musa_pkg7549/mtcc-nightly-x86_64-linux-gnu-ubuntu-20.04.tar.gz"

kmd_pkg_vps="kmd-vps_x86-mtgpu_linux-xorg-release-pdump_off.tar"
kmd_newest_pkg_release="kmd-sudi_x86-mtgpu_linux-xorg-release-pdump_off.tar"
kmd_stable_pkg_release="kmd-hw_x86-mtgpu_linux-xorg-release-pdump_off.tar"
umd_pkg_release="umd_x86-mtgpu_linux-xorg-release-pdump_off.tar"
murt_pkg_release="MUSA-Runtime_use_armory.tar.gz"

kmd_pkg_pdump="kmd-sudi_x86-mtgpu_linux-xorg-debug-pdump_on.tar"
umd_pkg_pdump="umd_x86-mtgpu_linux-xorg-debug-pdump_on.tar"
murt_pkg_pdump="MUSA-Runtime_debug_pdump.tar.gz"

# in defualt : the newest release version without pdump
ddk_path=${ddk_newest_path}
# musa_path=${musa_newest_path}
musa_path=${musa_history_path}
kmd_pkg=${kmd_newest_pkg_release}
umd_pkg=${umd_pkg_release}
murt_pkg=${murt_pkg_release}
mtcc_pkg=${mtcc_pkg_release}
tool_pkg=${tool_pkg_release}


echo_info() {
    echo -e "\033[33m"$1"\033[0m"
}

echo_success() {
    echo -e "\033[32m"$1"\033[0m"
}

echo_error() {
    echo -e "\033[31m"$1"\033[0m"
}

check_success(){
    ret=$1
    tag=${@:2}
    echo $tag
    if [ ${ret} -ne 0 ]; then
        echo -e "\033[31m- ${tag} failed \033[0m"
        echo -e "\033[33m- ${tag} please check, -h for help \033[0m"
        exit 1
    else
        echo -e "\033[32m- ${tag} successed \033[0m"
    fi
}

help() {
    echo_info "----------------------------------------------------------------"
    name="$(basename "$(realpath "${BASH_SOURCE:-$0}" )")"
    echo_info "Description:"
    echo_info "This script will update the musa environment automatically,"
    echo_info "including kmd, umd, musa-runtime, mtcc and musa_toolkit cmake."
    echo_info "It will download the pkg to \$download_dir, or install from local pkg."
    echo_info "The musa-runtime and mtcc will be installed to ${musa_install_dir}."
    echo_info "Usage:"
    echo_info " ${name} [-d] [-s|-n] [-k|-v] [-u] [-r] [-c] [-t]"
    echo_info "Details:"
    echo_info " -i : only install local pkg, without download"
    echo_info " -d : set pdump on (kmd umd musa-runtime)"
    echo_info " -s : download the stable package (only for ddk temporarily)"
    echo_info " -n : download the newest package - in default"
    echo_info " -v : install gr_kmd_dist_vps"
    echo_info " -k : install gr_kmd_dist_sudi"
    echo_info " -u : install gr_umd_dist"
    echo_info " -r : install musa-runtime"
    echo_info " -c : install mtcc package"
    echo_info " -t : install musa_toolkit (cmake/mtcc/runtime)"
    echo_info "----------------------------------------------------------------"
    echo_info "e.g."
    echo_info "${name} -kut # install kmd, umd, cmake, mtcc, and musa-runtime together"
    exit 0
}

clean_musa_install_dir() {
    [ -d "${musa_install_dir}" ] && rm -rf "${musa_install_dir}"
}

install_umd() {
    #install umd
    umd_dir="gr_umd_dist"
    mkdir -p "${download_dir}/${umd_dir}"
    pushd "${download_dir}/${umd_dir}"
    echo_info "${umd_pkg} installing..."
    if [ "${do_download_pkg}"x = "true"x ]; then
        [ -d "${umd_dir}" ] && rm -rf "${umd_dir}"
        [ -f "${umd_pkg}" ] && rm -rf "${umd_pkg}"
        rm -rf *.txt
        wget -P ./ "${ddk_path}/${umd_pkg}" --no-check-certificate
    fi
    if [ ! -f "${umd_pkg}" ]; then
        echo_error "${download_dir}/${umd_pkg} does not exist, please check!"
        exit 1
    fi
    [ ! -d "${umd_dir}" ] && tar -xvf ${umd_pkg}
    ${download_dir}/${umd_dir}/${umd_dir}/*mtgpu_linux*/install.sh -u || true
    rm -rf /usr/lib/`uname -m`-linux-gnu/musa
    cd ${download_dir}/${umd_dir}/${umd_dir}/*mtgpu_linux*/
    ./install.sh -s . && ldconfig /usr/lib/`uname -r`-linux-gnu/musa
    cd -
    if ! grep -q "/usr/lib/`uname -m`-linux-gnu/musa" /etc/ld.so.conf.d/`uname -m`-linux-gnu.conf; then
        sed -i "1i /usr/lib/`uname -m`-linux-gnu/musa" /etc/ld.so.conf.d/`uname -m`-linux-gnu.conf
        ldconfig /usr/lib/'uname -m'-linux-gnu/musa/
    fi
    check_success $? "${umd_pkg} install"
    echo_info "$(cat *.txt)"
    popd
}

install_kmd() {
    #install kmd
    kmd_dir="gr_kmd_dist"
    mkdir -p ${download_dir}/${kmd_dir}
    pushd ${download_dir}/${kmd_dir}
    echo_info "${kmd_pkg} installing..."
    if [ "${do_download_pkg}"x = "true"x ]; then
        [ -d "${kmd_dir}" ] && rm -rf "${kmd_dir}"
        [ -f "${kmd_pkg}" ] && rm -rf "${kmd_pkg}"
        rm -rf *.txt
        wget -P ./ "${ddk_path}/${kmd_pkg}" --no-check-certificate
    fi
    if [ ! -f "${kmd_pkg}" ]; then
        echo_error "${download_dir}/${kmd_pkg} does not exist, please check!"
        exit 1
    fi
    [ ! -d "${kmd_dir}" ] && tar -xvf "${kmd_pkg}"
    if lsmod | grep -q "^mt_peermem"; then
        rmmod mt_peermem || true
    fi
    if lsmod | grep -q "^snd_compress"; then
        rmmod snd-compress  || true
    fi
    if lsmod | grep -q "^snd_pcm_dmaengine"; then
        rmmod snd-pcm-dmaengine || true
    fi

    if lsmod | grep -q "^mtgpu"; then
        lightdm_status=$(systemctl is-active lightdm)
        if [ "$lightdm_status" = "active" ]
        then
            systemctl stop lightdm
            echo "waiting for lightdm closed"
            sleep 3
        fi
        rmmod mtgpu || true
    fi
    if lsmod | grep -q "^snd_pcm"; then
        rmmod snd-pcm || true
    fi
    if lsmod | grep -q "^snd_rawmidi"; then
        rmmod snd-seq-midi || true
        rmmod snd-seq-midi-event || true
        rmmod snd-rawmidi || true
    fi  
    if lsmod | grep -q "^snd_seq"; then
        rmmod snd-seq || true
        rmmod snd-seq-device || true
    fi
    if lsmod | grep -q "^snd"; then
        rmmod snd-timer || true
        rmmod snd || true
    fi
    if lsmod | grep -q "^soundcore"; then
        rmmod soundcore || true
    fi
    cd /usr/lib/modules/`uname -r`/kernel/sound
    insmod soundcore.ko || true
    cd -
    cd /usr/lib/modules/`uname -r`/kernel/sound/core
    insmod snd.ko || true
    insmod snd-timer.ko || true
    insmod snd-pcm.ko || true
    insmod snd-pcm-dmaengine.ko || true
    insmod snd-compress.ko || true
    cd -
    cd ${download_dir}/${kmd_dir}/${kmd_dir}/*mtgpu_linux*/lib/modules/5.4.0-42-generic/extra/
    kmd_system_dir="/lib/modules/`uname -r`/extra"
    if [ ! -d ${kmd_system_dir} ]; then
        mkdir ${kmd_system_dir}
    fi
    cp mtgpu.ko ${kmd_system_dir}
    modprobe drm
    modprobe drm_kms_helper
    # load module according /etc/modprobe.d/mtgpu.conf
    # modprobe mtgpu 
    # load module according explicit parameter setting 
    insmod mtgpu.ko disable_audio=1 display=none enable_large_mem_mode=1 disable_vpu=1
    echo "options mtgpu display=none disable_audio=1 enable_large_mem_mode=1 disable_vpu=1" > /etc/modprobe.d/mtgpu.conf
    cd -
    check_success $? "${kmd_pkg} install"
    echo_info "$(cat *.txt)"
    popd
}

install_murt() {
    #install musa_runtime
    [ ! -d "${musa_install_dir}" ] && mkdir -p "${musa_install_dir}"
    murt_dir="MUSA-Runtime"
    mkdir -p "${download_dir}/${murt_dir}"
    pushd "${download_dir}/${murt_dir}"
    echo_info "${murt_pkg} installing..."
    #FIXME: (lms) this is a version workaround for Global Event object destructor SEGV of the latest MUSARuntime.
    tmp_musart_path="https://oss.mthreads.com/release-ci/computeQA/musa/history/20231030MUSA-Runtime252780dd2compute_musa_pkg7029/MUSA-Runtime_use_armory.tar.gz"
    if [ "${do_download_pkg}"x = "true"x ]; then
        [ -d "${murt_dir}" ] && rm -rf "${murt_dir}"
        [ -f "${murt_pkg}" ] && rm -rf "${murt_pkg}"
        rm -rf *.txt
        # wget -P ./ "${musa_path}/${murt_pkg}" --no-check-certificate
        wget -P ./ ${tmp_musart_path} --no-check-certificate
    fi
    if [ ! -f "${murt_pkg}" ]; then
        echo_error "${download_dir}/${murt_pkg} does not exist, please check!"
        exit 1
    fi
    [ ! -d "${murt_dir}" ] && tar -xvzf "${murt_pkg}"
    cd ${download_dir}/${murt_dir}/${murt_dir}
    ./install.sh "${musa_install_dir}"
    echo 1048576 >> /proc/sys/fs/file-max
    sysctl -p
    echo "fs.file-max = 1048576" >> /etc/sysctl.conf
    echo "* soft nofile 1048576" >> /etc/security/limits.conf
    echo "* hard nofile 1048576" >> /etc/security/limits.conf
    check_success $? "${murt_pkg} install"
    cd -
    echo_info "$(cat *.txt)"
    popd
}

install_mtcc(){
    wget -P ./ "${musa_path}/${mtcc_pkg}" --no-check-certificate
    mkdir -p mtcc
    tar -xvf mtcc-nightly-x86_64-linux-gnu-ubuntu-20.04.tar.gz -C mtcc
    pushd ./mtcc
    bash install.sh
    if [ $? -eq 0 ]; then
        echo -e "\033[31minstall mtcc success!! \033[0m"
    else
        echo -e "\033[31minstall mtcc failed!! \033[0m"
    fi
    popd
}
install_tool() {
    wget -P ./ "${musa_path}/${tool_pkg}" --no-check-certificate
    tar -zxvf musa_toolkits_install.tar.gz
    bash ./musa_toolkits_install/install.sh 
    if [ $? -eq 0 ]; then
        echo -e "\033[31minstall musa toolkits success!! \033[0m"
    else
        echo -e "\033[31minstall musa toolkits failed!! \033[0m"
    fi
}

main() {
    #[ "${do_install_murt}" = "true" ] || [ "${do_install_mtcc}" = "true" ] && clean_musa_install_dir
    [ ! -d "${download_dir}" ] && mkdir -p "${download_dir}"
    [ "${do_install_umd}" = "true" ] && install_umd
    [ "${do_install_kmd}" = "true" ] && install_kmd
    [ "${do_install_tool}" = "true" ] && install_tool
    [ "${do_install_mtcc}" = "true" ] && install_mtcc
    [ "${do_install_murt}" = "true" ] && install_murt
}


while getopts 'isndvkurctgh' OPT; do
    case $OPT in
        i) 
            do_download_pkg="false"
            ;;
        s) 
            ddk_path=${ddk_stable_path}
	    # musa_stable is not available
            musa_path=${musa_newest_path}
            kmd_pkg=${kmd_stable_pkg_release}
            ;;
        n) 
            ddk_path=${ddk_newest_path}
            musa_path=${musa_newest_path}
            ;;
        d) 
            kmd_pkg=${kmd_pkg_pdump}
            umd_pkg=${umd_pkg_pdump}
            murt_pkg=${murt_pkg_pdump}
            #mtcc_pkg=${mtcc_pkg_pdump}
            ;;
        v) 
            kmd_pkg=${kmd_pkg_vps}
            do_install_kmd="true"
            ;;
        k) 
            do_install_kmd="true"
            ;;
        u) 
            do_install_umd="true"
            ;;
        r) 
            do_install_murt="true"
            ;;
        c) 
            do_install_mtcc="true"
            ;;
        t) 
            do_install_tool="true"
            ;;
        h) help;;
        ?) help;;
    esac
done

main