#!/bin/bash
set -e

echo -e "\033[0;33m===========================\033[0m"
echo -e "\033[0;33mCreating directory pretrained/ for pretrained models...\033[0m"
mkdir -p pretrained/

echo -e "\033[0;33mDownloading pretrained models from huggingface using huggingface-cli. If you do not have this command, please install using 'pip install \"huggingface_hub[cli]\"'.\033[0m"
echo -e "\033[0;33mIf you do not have access to huggingface, we also support modelscope.\033[0m"
echo -e "\033[0;33mPlease open this script and change the next command to modelscope. It will require you to have run 'pip install modelscope'.\033[0m"
echo -e "\033[0;33m===========================\033[0m"

# Normally, we use the huggingface-cli to download the models.
huggingface-cli download cantabile-kwok/lscodec_50hz --local-dir pretrained/
# If you want to use modelscope instead, uncomment the following line:
# modelscope download --model CantabileKwok/lscodec-50hz --local_dir pretrained/

echo -e "\033[0;33m===========================\033[0m"
echo -e "\033[0;33mFinished downloading pretrained LSCodec models.\033[0m"
echo -e "\033[0;33mFor the WavLM model, as the official link is Google drive, please download it manually from the link below and place it in the pretrained/ directory.\033[0m"
echo -e "\033[0;33mhttps://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view?usp=share_link\033[0m"
echo -e "\033[0;33m===========================\033[0m"
