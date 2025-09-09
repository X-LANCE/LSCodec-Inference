#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# MIT License
# Copyright (c) 2025 Yiwei Guo
# This file is based on ParallelWaveGAN (https://github.com/kan-bayashi/ParallelWaveGAN). 
# Above is the original copyright notice.


"""Decode with trained Parallel WaveGAN Generator."""

import argparse
import logging
from kaldiio import WriteHelper
from pathlib import Path
import os
import time
import re

import numpy as np
import librosa

import soundfile as sf
import torch
import yaml

from tqdm import tqdm
import kaldiio
from lscodec.ssl_models.wavlm_extractor import Extractor as WavLMExtractor
from lscodec.utils import load_model, load_vocoder, load_audio

def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained VQVAE "
    )
    parser.add_argument(
        "--wav-scp",
        "--scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file. "
        "you need to specify either wav-scp or dumpdir.",
    )
    parser.add_argument(
        "--prompt-wav-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file. "
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    # parser.add_argument(
    #     "--encoder-checkpoint",
    #     type=str,
    #     default="pretrained/lscodec_encoder.pt",
    #     help="checkpoint file to be loaded.",
    # )
    # parser.add_argument(
    #     "--encoder-config",
    #     default='pretrained/encoder_config.yml',
    #     type=str,
    #     help="yaml format configuration file. if not explicitly provided, "
    #     "it will be searched in the checkpoint directory. (default=None)",
    # )
    # parser.add_argument(
    #     "--vocoder-checkpoint",
    #     type=str,
    #     default="pretrained/lscodec_vocoder.pt",
    #     help="checkpoint file to be loaded.",
    # )
    # parser.add_argument(
    #     "--vocoder-config",
    #     default='pretrained/vocoder_config.yml',
    #     type=str,
    #     help="yaml format configuration file. if not explicitly provided, "
    #     "it will be searched in the checkpoint directory. (default=None)",
    # )
    # parser.add_argument(
    #     "--wavlm-path",
    #     type=str,
    #     default="pretrained/WavLM-Large.pt",
    #     help="path to WavLM checkpoint.",
    # )
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        default="pretrained/",
        help="directory to save pretrained model. (default=None). It will need to contain WavLM-Large.pt, lscodec_encoder.pt, lscodec_vocoder.pt, encoder_config.yml, vocoder_config.yml and codebook.npy",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load encoder config
    encoder_config_path = os.path.join(args.pretrained_dir, "encoder_config.yml")
    encoder_checkpoint_path = os.path.join(args.pretrained_dir, "lscodec_encoder.pt")
    with open(encoder_config_path) as f:
        encoder_config = yaml.load(f, Loader=yaml.Loader)
    encoder_config.update(vars(args))
    encoder_config['pretrain_codebook'] = os.path.join(args.pretrained_dir, "codebook.npy")  # overwrite
    
    # load vocoder config
    vocoder_config_path = os.path.join(args.pretrained_dir, "vocoder_config.yml")
    vocoder_checkpoint_path = os.path.join(args.pretrained_dir, "lscodec_vocoder.pt")
    with open(vocoder_config_path) as f:
        vocoder_config = yaml.load(f, Loader=yaml.Loader)
    vocoder_config.update(vars(args))
    vocoder_config['vq_codebook'] = os.path.join(args.pretrained_dir, "codebook.npy")  # overwrite

    # check arguments
    if args.wav_scp is None or args.prompt_wav_scp is None:
        raise ValueError("Please specify --wav-scp and --prompt-wav-scp.")

    utt2path = dict()
    with open(args.wav_scp, 'r') as fr:
        for line in fr.readlines():
            terms = line.strip().split()
            if len(terms) != 2:
                raise ValueError(f"Invalid line in wav_scp: {line.strip()}")
            utt2path[terms[0]] = terms[1]
    prompt_scp = dict()
    with open(args.prompt_wav_scp, 'r') as fr:
        for line in fr.readlines():
            terms = line.strip().split()
            if len(terms) != 2:
                raise ValueError(f"Invalid line in prompt_wav_scp: {line.strip()}")
            utt_id, audio_path = terms
            prompt_scp[utt_id] = audio_path

    logging.info(f"The number of features to be decoded = {len(utt2path)}.")

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using GPU.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")
    model = load_model(encoder_config, encoder_checkpoint_path)
    logging.info(f"Loaded encoder parameters from {encoder_checkpoint_path}.")
    model = model.eval().to(device)

    vocoder = load_vocoder(vocoder_config, vocoder_checkpoint_path)
    logging.info(f"Loaded vocoder parameters from {vocoder_checkpoint_path}.")
    vocoder = vocoder.eval().to(device)
    # load vq codebook
    feat_codebook = torch.tensor(np.load(vocoder_config["vq_codebook"], allow_pickle=True)).to(device)  # (V, D)
    if feat_codebook.ndim == 2:
        feat_codebook = feat_codebook.unsqueeze(0)
    feat_codebook_numgroups = feat_codebook.shape[0]
    feat_codebook = torch.nn.ModuleList([torch.nn.Embedding.from_pretrained(feat_codebook[i], freeze=True) for i in range(feat_codebook_numgroups)])

    wavlm_path = os.path.join(args.pretrained_dir, "WavLM-Large.pt")
    wavlm = WavLMExtractor(checkpoint=wavlm_path, device=device)

    # start generation
    with torch.no_grad(), tqdm(utt2path.items(), desc="[prompted recon]") as pbar:
        for idx, (utt_id, audio_path) in enumerate(pbar, 1):
            prompt_path = prompt_scp[utt_id]
            audio, sr = load_audio(audio_path)
            prompt_audio, prompt_sr = load_audio(prompt_path)

            if sr != encoder_config['feature_extractor_params']['fs']:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=encoder_config['feature_extractor_params']['fs'])

            if prompt_sr != 16000:
                prompt_audio = librosa.resample(prompt_audio, orig_sr=prompt_sr, target_sr=16000)

            audio = torch.tensor(audio).float().to(device)  # (L)
            audio = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, L)

            prompt = wavlm.extract(prompt_audio)
            prompt = prompt.float().unsqueeze(0).to(device) # [1, L, D]

            # generate
            _, _, embed_index = model.encode(audio)
            if vocoder_config.get("repeat_input_tokens", False):
                # in LSCodec-25Hz, we still trained the vocoder on 50Hz input, so repeat_interleave is needed.
                embed_index = embed_index.repeat_interleave(2, dim=0)
            vqvec = torch.cat([feat_codebook[i](embed_index[:, i]) for i in range(feat_codebook_numgroups)], dim=-1).unsqueeze(0)  # (1, L, D)
            y = vocoder.inference(vqvec, prompt)[-1].view(-1)

            tgt_dir = os.path.dirname(os.path.join(args.outdir, f"{utt_id}.wav"))
            os.makedirs(tgt_dir, exist_ok=True)
            basename = os.path.basename(f"{utt_id}.wav")
            # save as PCM 16 bit wav file
            sf.write(
                os.path.join(tgt_dir, basename),
                y.cpu().numpy(),
                vocoder_config["sampling_rate"],
                "PCM_16",
            )

    # report average RTF
    logging.info(
        f"Finished generation of {idx} utterances."
    )


if __name__ == "__main__":
    main()
