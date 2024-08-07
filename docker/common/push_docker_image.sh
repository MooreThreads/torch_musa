#!/bin/bash
USER=""
PWD=""
LOCAL_IMAGE_NAME=""
LOCAL_IMAGE_TAG="latest"
REMOTE_IMAGE_NAME="sh-harbor.mthreads.com/mt-ai/musa-pytorch-dev"
REMOTE_IMAGE_TAG=""
REMOTE_URL="https://sh-harbor.mthreads.com/"

usage() {
  echo -e "\033[1;32mThis script is used for pushing finished image to the remote automatically. \033[0m"
  echo -e "\033[1;32mParameters usage: \033[0m"
  echo -e "\033[32m    -u/--user                : The username of remote repository. \033[0m"
  echo -e "\033[32m    -p/--pwd                 : The password of remote repository. \033[0m"
  echo -e "\033[32m    --local_image_name       : Means docker image name locally. \033[0m"
  echo -e "\033[32m    --local_image_tag        : Means docker image tag locally. \033[0m"
  echo -e "\033[32m    --remote_image_name      : Means docker image name remotely. \033[0m"
  echo -e "\033[32m    --remote_image_tag       : Means docker image tag remotely. \033[0m"
  echo -e "\033[32m    --remote_url             : Means the url of remote repository. \033[0m"
  echo -e "\033[32m    -h/--help                : Help information. \033[0m"
}
 
# parse paremters
parameters=`getopt -o +u:p:h --long user:,pwd:,local_image_name:,local_image_tag:,remote_image_name:,remote_image_tag:,remote_url:,help, -n "$0" -- "$@"`
[ $? -ne 0 ] && { echo "Try '$0 --help' for more information."; exit 1; }
 
eval set -- "$parameters"

while true;do
  case "$1" in
    -u|--user) USER=$2; shift 2;;  
    -p|--pwd) PWD=$2; shift 2;;
    --local_image_name) LOCAL_IMAGE_NAME=$2; shift 2;;
    --local_image_tag) LOCAL_IMAGE_TAG=$2; shift 2;;
    --remote_image_name) REMOTE_IMAGE_NAME=$2; shift 2;;
    --remote_image_tag) REMOTE_IMAGE_TAG=$2; shift 2;;
    --remote_url) REMOTE_URL=$2; shift 2;;
    -h|--help) usage; exit ;;
    --) shift ; break ;;
    *) usage;exit 1;;
esac
done

sudo docker login -u $USER -p $PWD $REMOTE_URL
sudo docker tag $LOCAL_IMAGE_NAME:$LOCAL_IMAGE_TAG $REMOTE_IMAGE_NAME:$REMOTE_IMAGE_TAG
sudo docker push $REMOTE_IMAGE_NAME:$REMOTE_IMAGE_TAG
