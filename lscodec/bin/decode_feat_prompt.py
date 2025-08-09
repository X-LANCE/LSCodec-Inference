#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# MIT License
# Copyright (c) 2025 Yiwei Guo
# This file is based on ParallelWaveGAN (https://github.com/kan-bayashi/ParallelWaveGAN). 
# Above is the original copyright notice.


"""Decode with Trained LSCodec Vocoder."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml
import kaldiio
from tqdm import tqdm

from lscodec.utils import load_vocoder


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode from LSCodec tokens to waveforms"
    )
    parser.add_argument(
        "--feats-scp",
        "--scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file. "
        "you need to specify either feats-scp or dumpdir.",
    )
    parser.add_argument(
        "--prompt-scp", 
        default=None, 
        type=str
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
        default='pretrained/lscodec_vocoder.pt',
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default='pretrained/vocoder_config.yml',
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
    if args.feats_scp is None:
        raise ValueError("Please specify --feats-scp.")

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using GPU.")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU.")
    model = load_vocoder(checkpoint=args.checkpoint, config=config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    model.backend.remove_weight_norm()
    model = model.eval().to(device)

    # load vq codebook
    feat_codebook = torch.tensor(np.load(config["vq_codebook"], allow_pickle=True)).to(device)  # (V, D)
    if feat_codebook.ndim == 2:
        feat_codebook = feat_codebook.unsqueeze(0)
    feat_codebook_numgroups = feat_codebook.shape[0]
    feat_codebook = torch.nn.ModuleList([torch.nn.Embedding.from_pretrained(feat_codebook[i], freeze=True) for i in range(feat_codebook_numgroups)])

    # start generation
    total_rtf = 0.0
    feats_scp = kaldiio.load_scp(args.feats_scp)
    prompt_scp = kaldiio.load_scp(args.prompt_scp)
    total_cnt = len(list(feats_scp.keys()))
    logging.info(f"The number of features to be decoded = {total_cnt}.")

    with torch.no_grad(), tqdm(feats_scp, desc="[decode]", total=total_cnt) as pbar:
        for idx, utt_id in enumerate(pbar, 1):
            c = feats_scp[utt_id]
            prompt = prompt_scp[utt_id]

            c = torch.tensor(c).to(device)  # (L, D)
            prompt = torch.tensor(prompt).unsqueeze(0).to(device)  # (1, L', D')

            vqidx = c.long()
            vqvec = torch.cat([feat_codebook[i](vqidx[:, i]) for i in range(feat_codebook_numgroups)], dim=-1).unsqueeze(0)  # (1, L, D)

            # generate
            start = time.time()
            y = model.inference(vqvec, prompt)[-1].view(-1)
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            tgt_dir = os.path.dirname(os.path.join(config["outdir"], f"{utt_id}.wav"))
            os.makedirs(tgt_dir, exist_ok=True)
            basename = os.path.basename(f"{utt_id}.wav")
            # save as PCM 16 bit wav file
            sf.write(
                os.path.join(tgt_dir, basename),
                y.cpu().numpy(),
                config["sampling_rate"],
                "PCM_16",
            )

    # report average RTF
    logging.info(
        f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
    )


if __name__ == "__main__":
    main()
