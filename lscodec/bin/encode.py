#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# MIT License
# Copyright (c) 2025 Yiwei Guo
# This file is based on ParallelWaveGAN (https://github.com/kan-bayashi/ParallelWaveGAN). 
# Above is the original copyright notice.


"""Encode with Trained LSCodec Encoder."""

import argparse
import logging
import os
import time

import numpy as np
import re
import librosa

import soundfile as sf
import torch
from pathlib import Path
import yaml

from tqdm import tqdm
import kaldiio

from lscodec.utils import load_model, load_audio
from kaldiio import WriteHelper


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Encode features with trained LSCodec"
    )
    parser.add_argument(
        "--wav-scp",
        "--scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file. "
        "you need to specify wav-scp.",
    )
    parser.add_argument(
        "--num-frames",
        default=None,
        type=str
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='pretrained/lscodec_encoder.pt',
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default='pretrained/encoder_config.yml',
        type=str,
        help="yaml format configuration file."
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

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if args.wav_scp is None:
        raise ValueError("Please specify --wav-scp.")

    utt2path = dict()
    with open(args.wav_scp, 'r') as fr:
        for line in fr.readlines():
            terms = line.strip().split()
            utt2path[terms[0]] = terms[1]
    logging.info(f"The number of features to be decoded = {len(utt2path)}.")

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using GPU.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")
    model = load_model(config, args.checkpoint)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    model = model.eval().to(device)

    # start generation
    outdir = str(Path(args.outdir).absolute())
    with torch.no_grad(), tqdm(utt2path.items(), desc="[encode]") as pbar, WriteHelper(f"ark,scp:{outdir}/feats.ark,{outdir}/feats.scp") as writer:
        for i, (utt_id, audio_path) in enumerate(pbar, 1):
            audio, sr = load_audio(audio_path)

            if sr != config['feature_extractor_params']['fs']:
                audio_for_extractor = librosa.resample(audio, orig_sr=sr, target_sr=config['feature_extractor_params']['fs'])
            else:
                audio_for_extractor = audio
            audio_for_extractor = torch.tensor(audio_for_extractor).float().to(device)  # (L)
            audio_for_extractor = audio_for_extractor.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
            means, _, embed_index = model.encode(audio_for_extractor)

            # now the embed_index is [L, 1] token tensor
            if embed_index is not None:
                writer(utt_id, embed_index.float().cpu().numpy())
            else:
                writer(utt_id, means.squeeze(0).cpu().numpy())


if __name__ == "__main__":
    main()
