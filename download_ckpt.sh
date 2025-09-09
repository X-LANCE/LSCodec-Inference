#!/bin/bash
set -e

VERSION=$1
if [ -z "$VERSION" ]; then
  VERSION="50hz"
fi
# if VERSION is not in '50hz' or '25hz', throw error
if [ "$VERSION" != "50hz" ] && [ "$VERSION" != "25hz" ]; then
  echo -e "\033[0;31mError: VERSION must be either '50hz' or '25hz'.\033[0m"
  exit 1
fi

# if VERSION is 50hz, target dir is pretrained; otherwise pretrained_25hz/
if [ "$VERSION" == "50hz" ]; then
  TARGET_DIR="pretrained/"
else
  TARGET_DIR="pretrained_25hz/"
fi

echo -e "\033[0;33m===========================\033[0m"
echo -e "\033[0;33mCreating directory $TARGET_DIR for pretrained $VERSION models...\033[0m"
mkdir -p $TARGET_DIR

echo -e "\033[0;33mDownloading pretrained models from huggingface using huggingface-cli. If you do not have this command, please install using 'pip install \"huggingface_hub[cli]\"'.\033[0m"
echo -e "\033[0;33mIf you do not have access to huggingface, we also support modelscope.\033[0m"
echo -e "\033[0;33mPlease open this script and change the next command to modelscope. It will require you to have run 'pip install modelscope'.\033[0m"
echo -e "\033[0;33m===========================\033[0m"

# Normally, we use the huggingface-cli to download the models.
if [ "$VERSION" == "50hz" ]; then
    huggingface-cli download cantabile-kwok/lscodec_50hz --local-dir $TARGET_DIR
else
    huggingface-cli download cantabile-kwok/lscodec_25hz --local-dir $TARGET_DIR
fi

# If you want to use modelscope instead, uncomment the following lines:
# if [ "$VERSION" == "50hz" ]; then
#     modelscope download --model CantabileKwok/lscodec-50hz --local_dir $TARGET_DIR
# else
#     modelscope download --model CantabileKwok/lscodec-25hz --local_dir $TARGET_DIR
# fi

echo -e "\033[0;33m===========================\033[0m"
echo -e "\033[0;33mFinished downloading pretrained LSCodec ($VERSION) models.\033[0m"
echo -e "\033[0;33mFor the WavLM model, as the official link is Google drive, please download it manually from the link below and place it in the $TARGET_DIR directory.\033[0m"
echo -e "\033[0;33mhttps://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view?usp=share_link\033[0m"
echo -e "\033[0;33m===========================\033[0m"
