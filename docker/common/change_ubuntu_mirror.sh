#! /bin/bash
set -ex

# maybe need to replace apt-get source
mv /etc/apt/sources.list /etc/apt/sources_backup.list
# major=$(echo $UBUNTU_VERSION | cut -d'.' -f1)
# case $major in
#   16) code_name="xenial" ;;
#   18) code_name="bionic" ;;
#   20) code_name="focal"  ;;
#   22) code_name="jammy"  ;;
# esac
codename=$(awk -F= '$1=="VERSION_CODENAME" {print $2}' /etc/os-release)
echo "deb http://mirrors.ustc.edu.cn/ubuntu/ $codename main restricted universe multiverse " >> /etc/apt/sources.list
echo "deb http://mirrors.ustc.edu.cn/ubuntu/ $codename-updates main restricted universe multiverse " >> /etc/apt/sources.list
echo "deb http://mirrors.ustc.edu.cn/ubuntu/ $codename-backports main restricted universe multiverse " >> /etc/apt/sources.list
echo "deb http://mirrors.ustc.edu.cn/ubuntu/ $codename-security main restricted universe multiverse " >> /etc/apt/sources.list
