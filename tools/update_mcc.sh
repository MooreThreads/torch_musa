set -exo pipefail
pushd /tmp
mkdir mtcc_update
pushd /tmp/mtcc_update
wget --no-check-certificate https://oss.mthreads.com/mt-ai-data/tmp/mtcc/1218/mtcc-x86_64-linux-gnu-ubuntu.tar.gz
tar -zxf mtcc-x86_64-linux-gnu-ubuntu.tar.gz
bash install.sh
popd
rm -rf mtcc_update
popd
echo "mcc update success"