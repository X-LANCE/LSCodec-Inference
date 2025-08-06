#!/bin/bash
set -e

echo -e "\033[0;33m===========================\033[0m"
echo -e "\033[0;33mCreating directory pretrained/ for pretrained models...\033[0m"
mkdir -p pretrained/

echo -e "\033[0;33mDownloading pretrained models from huggingface using wget...\033[0m"
echo -e "\033[0;33mIf this fails, please download manually from the links below and place them in the pretrained/ directory.\033[0m"
echo -e "\033[0;33m===========================\033[0m"

wget -O pretrained/codebook.npy https://huggingface.co/cantabile-kwok/lscodec_50hz/resolve/main/codebook.npy?download=true
wget -O pretrained/lscodec_encoder.pt https://huggingface.co/cantabile-kwok/lscodec_50hz/resolve/main/lscodec_encoder.pt?download=true
wget -O pretrained/lscodec_vocoder.pt https://huggingface.co/cantabile-kwok/lscodec_50hz/resolve/main/lscodec_vocoder.pt?download=true
wget -O pretrained/encoder_config.yml https://huggingface.co/cantabile-kwok/lscodec_50hz/resolve/main/encoder_config.yml?download=true
wget -O pretrained/vocoder_config.yml https://huggingface.co/cantabile-kwok/lscodec_50hz/resolve/main/vocoder_config.yml?download=true

echo -e "\033[0;33m===========================\033[0m"
echo -e "\033[0;33mFinished downloading pretrained LSCodec models.\033[0m"
echo -e "\033[0;33mFor the WavLM model, as the official link is Google drive, please download it manually from the link below and place it in the pretrained/ directory.\033[0m"
echo -e "\033[0;33mhttps://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view?usp=share_link\033[0m"
echo -e "\033[0;33m===========================\033[0m"