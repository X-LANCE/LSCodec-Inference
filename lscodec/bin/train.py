#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

# Portions of this code are a modification of the ParallelWaveGAN project.
# The original source code can be found at:
# https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/bin/train.py
#
# Copyright (c) 2025 Yiwei Guo
# Licensed under the MIT license.

import argparse
import logging
import os
import sys
import random

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import lscodec
import lscodec.models
from lscodec.models.quantization.core_vq import GroupVectorQuantization
import lscodec.optimizers
from torch.utils.data.distributed import DistributedSampler

from lscodec.datasets.scp_dataset import AudioMelSCPDataset
from lscodec.utils.espnet_utils import pad_list, make_non_pad_mask
from lscodec.datasets.perturbs import Perturbs

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


def KL_loss(means, logvars, mask):
    means = means.masked_select(mask)
    logvars = logvars.masked_select(mask)
    logvars = torch.clamp(logvars, -3, 3)  # NOTE: added to ensure stability
    loss = -0.5*logvars + 0.5*(logvars.exp() + means.pow(2)) - 0.5
    # if torch.isnan(loss).any():
        # logging.info("detected nan in kl_loss")
        # raise ValueError()
    # loss = torch.nan_to_num(loss, nan=10)
    return loss.mean()


class Trainer(object):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(
            self,
            steps,
            epochs,
            data_loader,
            sampler,
            model,
            optimizer,
            scheduler,
            config,
            device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contain "train" and "dev" loaders.
            model (dict): Dict of models. It must contain "generator" model.
            optimizer (dict): Dict of optimizers. It must contain "generator" and optimizer.
            scheduler (dict): Dict of schedulers. It must contain "generator" and schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
            }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(
                state_dict["model"]["generator"]
            )
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"]
            )
            self.scheduler["generator"].load_state_dict(
                state_dict["scheduler"]["generator"]
            )

    def load_vae_params(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        generator = state_dict['model']['generator']
        frontend_state_dict = {}
        backend_state_dict = {}
        for key in generator.keys():
            if key.startswith("frontend"):
                frontend_state_dict[".".join(key.split(".")[1:])] = generator[key]
            elif key.startswith("backend"):
                backend_state_dict[".".join(key.split(".")[1:])] = generator[key]
            else:
                raise NotImplementedError(f"Unknown key: {key}")
        if self.config["distributed"]:
            self.model['generator'].module.frontend.load_state_dict(frontend_state_dict)
            self.model['generator'].module.backend.load_state_dict(backend_state_dict)
        else:
            self.model['generator'].frontend.load_state_dict(frontend_state_dict)
            self.model['generator'].backend.load_state_dict(backend_state_dict)

    def open_codebook_expire(self):
        if self.config['distributed']:
            for group_i in range(len(self.model['generator'].module.quantizer.groups)):
                self.model['generator'].module.quantizer.groups[group_i]._codebook.perform_expire = True
        else:
            for group_i in range(len(self.model['generator'].quantizer.groups)):
                self.model['generator'].quantizer.groups[group_i]._codebook.perform_expire = True

    def open_codebook_ema(self):
        if self.config['distributed']:
            for group_i in range(len(self.model['generator'].module.quantizer.groups)):
                self.model['generator'].module.quantizer.groups[group_i]._codebook.perform_ema_update = True
        else:
            for group_i in range(len(self.model['generator'].quantizer.groups)):
                self.model['generator'].quantizer.groups[group_i]._codebook.perform_ema_update = True

    def _train_step(self, batch):
        """Train model one step."""
        if self.config['have_quantizer'] and self.config['quantizer_class'] == "KMeans":
            if self.steps > self.config['quantizer_params']['perform_expire_after_steps']:
                self.open_codebook_expire()
            if self.steps > self.config['quantizer_params']['perform_ema_after_steps']:
                self.open_codebook_ema()
        if self.config['have_quantizer'] and self.config['quantizer_class'] == "Gumbel":
            self.model['generator'].module.quantizer.set_num_updates(self.steps)

        # parse batch
        aux, target, idx, prompt, y_for_extractor, xlens, prompt_lens = batch
        target = target.to(self.device)
        prompt = prompt.to(self.device)
        idx = idx.to(self.device)
        y_for_extractor = y_for_extractor.unsqueeze(-2).to(self.device)  # (B, 1, T')

        # build mask
        mask = make_non_pad_mask(xlens).to(self.device)  # (B, L) True for valid values
        prompt_mask = make_non_pad_mask(prompt_lens).to(self.device)  # (B, L_prompt)

        #######################
        #      Generator      #
        #######################
        if self.config['distributed']:
            tolerance = self.model['generator'].module.length_mismatch_tolerance
        else:
            tolerance = self.model['generator'].length_mismatch_tolerance
        target_, idx_logits_, _, commitment_loss, mask, means, logvars = self.model["generator"](y_for_extractor, prompt, mask, prompt_mask)
        # (B, L, 80)

        if target.shape[1] != mask.shape[-1]:
            if abs(target.shape[1] - mask.shape[-1]) <= tolerance:
                min_len = min(target.shape[1], mask.shape[-1])
                target = target[:, :min_len, :]
                mask = mask[..., :min_len]
                target_ = target_[:, :min_len, :]
                idx = idx[:, :min_len, :]
                idx_logits_ = idx_logits_[:, :min_len, :]

        # initialize
        gen_loss = 0.0
        # frontend target prediction loss
        frontend_target_pred_loss = F.l1_loss(torch.masked_select(target, mask.unsqueeze(-1)),
                                            torch.masked_select(target_, mask.unsqueeze(-1)))
        self.total_train_loss["train/frontend_target_pred_loss"] += frontend_target_pred_loss.item()
        gen_loss += self.config["lambda_frontend_target_prediction"] * frontend_target_pred_loss
        
        idx_criterion = torch.nn.CrossEntropyLoss()
        V = idx_logits_.shape[-1]
        frontend_idx_pred_loss = idx_criterion(torch.masked_select(idx_logits_, mask.unsqueeze(-1)).view(-1, V),
                                               torch.masked_select(idx.squeeze(-1), mask))
        self.total_train_loss["train/frontend_idx_pred_loss"] += frontend_idx_pred_loss.item()
        gen_loss += self.config["lambda_frontend_idx_prediction"] * frontend_idx_pred_loss

        if commitment_loss is None:
            if self.config['quantizer_class'] != "Gumbel":  # In this case, no additional loss.
                # VQVAE KL loss
                kl_loss = KL_loss(means, logvars, mask.unsqueeze(-1))
                gen_loss += self.config['lambda_kl_loss'] * kl_loss
                self.total_train_loss['train/kl_loss'] += kl_loss.item()
        else:
            # VQ loss (if kmeans)
            if self.config['quantizer_class'] == "KMeans":
                gen_loss += self.config['lambda_kmeans_loss'] * commitment_loss
                self.total_train_loss['train/kmeans_loss'] += commitment_loss.item()

        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        # gen_loss = torch.nan_to_num(gen_loss, nan=100)  # NOTE: will this ensure stability?
        if torch.isnan(gen_loss):
            logging.info(f"Find NAN in loss. Refusing to update.")
            gen_loss = 0.
            for p in self.model['generator'].parameters():
                gen_loss = gen_loss + 0.0 * p.sum()
            # gen_loss = torch.sum([0.0*p.sum() for p in self.model['generator'].parameters()])  # fake loss. Gradient=0

        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"],
            )
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()
        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        aux, target, idx, prompt, y_for_extractor, xlens, prompt_lens = batch
        target = target.to(self.device)
        prompt = prompt.to(self.device)
        idx = idx.to(self.device)
        y_for_extractor = y_for_extractor.unsqueeze(-2).to(self.device)  # (B, 1, T')

        # build mask
        mask = make_non_pad_mask(xlens).to(self.device)  # (B, L)
        prompt_mask = make_non_pad_mask(prompt_lens).to(self.device)  # (B, L_prompt)

        #######################
        #      Generator      #
        #######################
        if self.config['distributed']:
            tolerance = self.model['generator'].module.length_mismatch_tolerance
        else:
            tolerance = self.model['generator'].length_mismatch_tolerance
        target_, idx_logits_, _, commitment_loss, mask, means, logvars = self.model["generator"](y_for_extractor, prompt, mask, prompt_mask)  # (B, L, 80), (B, C, T)
        if target.shape[1] != mask.shape[-1]:
            if abs(target.shape[1] - mask.shape[-1]) <= tolerance:
                min_len = min(target.shape[1], mask.shape[-1])
                target = target[:, :min_len, :]
                mask = mask[..., :min_len]
                target_ = target_[:, :min_len, :]
                idx = idx[:, :min_len, :]
                idx_logits_ = idx_logits_[:, :min_len, :]

        # frontend target prediction loss
        frontend_target_pred_loss = F.l1_loss(torch.masked_select(target, mask.unsqueeze(-1)),
                                            torch.masked_select(target_, mask.unsqueeze(-1)))
        self.total_eval_loss["eval/frontend_target_pred_loss"] += frontend_target_pred_loss.item()
        
        idx_criterion = torch.nn.CrossEntropyLoss()
        V = idx_logits_.shape[-1]
        frontend_idx_pred_loss = idx_criterion(torch.masked_select(idx_logits_, mask.unsqueeze(-1)).view(-1, V),
                                               torch.masked_select(idx.squeeze(-1), mask))
        self.total_eval_loss["eval/frontend_idx_pred_loss"] += frontend_idx_pred_loss.item()

        if commitment_loss is None:
            if self.config['quantizer_class'] != "Gumbel":  # In this case, no additional loss.
                # VAE KL loss
                kl_loss = KL_loss(means, logvars, mask.unsqueeze(-1))
                self.total_eval_loss['eval/kl_loss'] += kl_loss.item()
        else:
            # VQ loss (if kmeans)
            if self.config['quantizer_class'] == "KMeans":
                self.total_eval_loss['eval/kmeans_loss'] += commitment_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
                tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            if self.config['have_quantizer'] and self.config['quantizer_class'] == "Gumbel":
                logging.info(f"(Steps: {self.steps}) gumbel temperature = {self.model['generator'].module.quantizer.curr_temp}")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collator(object):
    """Customized collator for Pytorch DataLoader in training."""

    def __init__(
            self,
            hop_size_for_extractor=256,
            win_length_for_extractor=1024,
            sampling_rate=16000,
            n_mel=80,
            force_from_start=False,
    ):
        """Initialize customized collator for PyTorch DataLoader.

        Args:
            hop_size_for_extractor (int): Hop size of features, in sampling points.
            win_length_for_extractor (int): window length of features.
            sampling_rate (int): sampling rate of waveform data
            n_mel (int): number of mel spectrogram dimensions
        """
        self.hop_size_for_extractor = hop_size_for_extractor
        self.win_length_for_extractor = win_length_for_extractor
        self.sampling_rate = sampling_rate
        self.n_mel = n_mel
        self.force_from_start = force_from_start

    def construct_prompt(self, mel_lens):
        prompt_lens = [random.randint(int(l / 3), int(l / 2)) for l in mel_lens]
        prompt_starts = []
        is_from_start = []
        for ml, pl in zip(mel_lens, prompt_lens):
            random_number = random.random()
            if self.force_from_start:
                random_number = 1
            if random_number > 0.5:
                # from start
                prompt_start = random.randint(0, 1 * self.sampling_rate//self.hop_size_for_extractor)
                is_from_start.append(True)
            else:
                # from ending
                prompt_start = random.randint((ml - 1 * self.sampling_rate//self.hop_size_for_extractor), ml) - pl
                is_from_start.append(False)
            prompt_starts.append(prompt_start)
        return prompt_lens, prompt_starts, is_from_start

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        This collator will automatically determine the prompt segment (acoustic context) for each utterance.
        The prompt is cut off from the current utterance, ranging from 2 to 3 seconds.
        The starting point of the prompt lies between the first 0-1 second range.
        Then, it parses the concatenated features into (3 dim auxiliary features, 2 dim VQ features, and 80 dim mel spectrograms)

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = T' * hop_size.
            Tensor: Target signal batch (B, 1, T).

        """
        batch = batch[0]

        # check length
        batch = [
            self._adjust_length(*b) for b in batch
        ]
        ys_for_extractor, mels, auxs, targets, idxs = ([b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch], [b[3] for b in batch], [b[4] for b in batch])

        batch_size = len(ys_for_extractor)
        prompt_lengths, prompt_starts, is_from_starts = self.construct_prompt([len(m) for m in mels])

        c_lengths = []

        prompts = torch.zeros(batch_size, max(prompt_lengths), self.n_mel)
        for i in range(batch_size):
            # data_offset = prompt_offsets[i] + prompt_lengths[i]
            prompts[i, :prompt_lengths[i]] = torch.tensor(mels[i][prompt_starts[i]:prompt_starts[i]+prompt_lengths[i], :])
            if is_from_starts[i]:
                targets[i] = targets[i][prompt_starts[i]+prompt_lengths[i]:]
                idxs[i] = idxs[i][prompt_starts[i]+prompt_lengths[i]:]
                auxs[i] = auxs[i][prompt_starts[i]+prompt_lengths[i]:]
                ys_for_extractor[i] = ys_for_extractor[i][(prompt_starts[i]+prompt_lengths[i])*self.hop_size_for_extractor:]
            else:
                targets[i] = targets[i][:prompt_starts[i]]
                idxs[i] = idxs[i][:prompt_starts[i]]
                auxs[i] = auxs[i][:prompt_starts[i]]
                ys_for_extractor[i] = ys_for_extractor[i][:prompt_starts[i]*self.hop_size_for_extractor]
            c_lengths.append(len(targets[i]))
        # NOTE: PAD_VALUES is set to 999 instead of 0, so that the code vectors to be quantized are not affected.
        targets = pad_list([torch.tensor(c) for c in targets], pad_value=0)  # (B, L, 80)
        idxs = pad_list([torch.tensor(c) for c in idxs], pad_value=0)  # (B, L, 80)
        idxs = idxs.long()
        auxs = pad_list([torch.tensor(c) for c in auxs], pad_value=0)  # (B, L, 3)
        ys_for_extractor = pad_list([torch.tensor(y, dtype=torch.float) for y in ys_for_extractor],
                                    pad_value=0)[:, :targets.size(1) * self.hop_size_for_extractor]
        assert targets.size(1) == auxs.size(1) == idxs.size(1)
        assert ys_for_extractor.size(1) == targets.size(1) * self.hop_size_for_extractor == auxs.size(1) * self.hop_size_for_extractor

        return auxs, targets, idxs, prompts, ys_for_extractor, c_lengths, prompt_lengths

    def _adjust_length(self, x, c, *args):
        """Adjust the audio and feature lengths.

        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.

        """
        if len(x) > len(c) * self.hop_size_for_extractor:
            x = x[(self.win_length_for_extractor - self.hop_size_for_extractor) // 2:]
            x = x[:len(c) * self.hop_size_for_extractor]

        # check the legnth is valid
        assert len(x) == len(c) * self.hop_size_for_extractor, f"{len(x)}, {len(c)}, {self.hop_size_for_extractor}"

        return x, c, *args


def main(rank, n_gpus):
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Parallel WaveGAN (See detail in ctx_vec2wav/bin/train.py)."
    )
    parser.add_argument(
        "--train-wav-scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file for training. "
    )
    parser.add_argument(
        "--train-idx-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for training. "
    )
    parser.add_argument(
        "--train-prompt-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for training. "
    )
    parser.add_argument(
        "--train-target-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for training. "
    )
    parser.add_argument(
        "--train-aux-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for training. "
    )
    parser.add_argument(
        "--train-segments",
        default=None,
        type=str,
        help="kaldi-style segments file for training.",
    )
    # parser.add_argument(
    #     "--train-xvector-scp",
    #     default=None,
    #     type=str,
    #     help="kaldi-style xvector.scp file for training.",
    # )
    parser.add_argument(
        "--train-num-frames",
        default=None,
        type=str,
        help="kaldi-style utt2num_frames file for training.",
    )
    parser.add_argument(
        "--train-utt2spk",
        default=None,
        type=str,
        help="kaldi-style utt2spk file for training.",
    )
    parser.add_argument(
        "--train-spk2gender",
        default=None,
        type=str,
        help="kaldi-style spk2gender file for training.",
    )
    parser.add_argument(
        "--dev-wav-scp",
        default=None,
        type=str,
        help="kaldi-style wav.scp file for validation. "
    )
    parser.add_argument(
        "--dev-idx-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for vaidation. "
    )
    parser.add_argument(
        "--dev-prompt-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for vaidation. "
    )
    parser.add_argument(
        "--dev-target-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for vaidation. "
    )
    parser.add_argument(
        "--dev-aux-scp",
        default=None,
        type=str,
        help="kaldi-style feats.scp file for vaidation. "
    )
    parser.add_argument(
        "--dev-segments",
        default=None,
        type=str,
        help="kaldi-style segments file for validation.",
    )
    # parser.add_argument(
    #     "--dev-xvector-scp",
    #     default=None,
    #     type=str,
    #     help="kaldi-style xvector.scp file for validation.",
    # )
    parser.add_argument(
        "--dev-num-frames",
        default=None,
        type=str,
        help="kaldi-style utt2num_frames file for validation.",
    )
    parser.add_argument(
        "--dev-utt2spk",
        default=None,
        type=str,
        help="kaldi-style utt2spk file for training.",
    )
    parser.add_argument(
        "--dev-spk2gender",
        default=None,
        type=str,
        help="kaldi-style spk2gender file for training.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--pretrain",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to load pretrained params. (default="")',
    )
    # parser.add_argument("--pretrain-feature-extractor",
    #                     default="",
    #                     type=str,
    #                     help="checkpoint file path to load pretrained params for CNN+trans feature extractor. Default=''")
    parser.add_argument("--pretrain-vae",
                        default="",
                        type=str,
                        help="pretrained VAE (no VQ)")
    parser.add_argument("--pretrain-codebook",
                        default="",
                        type=str,
                        help="checkpoint file path to load pretrained params for codebook. Default=''")
    # parser.add_argument("--pretrain-ctx-vec2wav",
    #                     default="",
    #                     type=str,
    #                     help="checkpoint file path to load pretrained params for ctx-vec2wav. Default=''")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    # parser.add_argument("--vq-codebook", default=None, type=str)
    parser.add_argument("--sampling-rate", type=int)
    parser.add_argument("--num-mels", type=int)
    parser.add_argument("--hop-size", type=int)
    parser.add_argument("--win-length", type=int)
    args = parser.parse_args()

    # init distributed training
    device = torch.device("cuda")
    # effective when using fixed size inputs
    # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.backends.cudnn.benchmark = True
    # setup for distributed training
    # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
    if n_gpus == 1:
        assert rank == 0

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # init process group
    logging.info("Synchronizing between all workers.")
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)
    logging.info("Finished init process group.")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config['rank'] = rank
    config['distributed'] = True
    config['world_size'] = n_gpus
    # config["version"] = lscodec.__version__  # add version info
    if rank == 0:
        with open(os.path.join(args.outdir, "config.yml"), "w") as f:
            yaml.dump(config, f, Dumper=yaml.Dumper)
        for key, value in config.items():
            logging.info(f"{key} = {value}")
    
    # get perturber
    perturber = Perturbs(
        formant_f0_perturb_params=config.get("formant_f0_perturb_params", dict()),
        speed_perturb_params=config.get("speed_perturb_params", dict()),
        speed_perturb_prob=config.get("speed_perturb_prob", 1.0)
    )

    # get dataset
    train_dataset = AudioMelSCPDataset(
        wav_scp=args.train_wav_scp,
        prompt_scp=args.train_prompt_scp,
        target_scp=args.train_target_scp,
        aux_scp=args.train_aux_scp,
        idx_scp=args.train_idx_scp,
        utt2num_frames=args.train_num_frames,
        segments=args.train_segments,
        batch_frames=config.get("batch_frames", None),
        batch_size=config.get("batch_size", None),
        min_num_frames=config.get("min_num_frames", None),
        max_num_frames=config.get("max_num_frames", None),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
        length_tolerance=config.get("length_tolerance", 2),
        prompt_min_length=config.get("prompt_min_length", 500),
        extractor_fs=config['feature_extractor_params'].get("fs", 16000),
        # speed_alpha_range=eval(config['speed_alpha_range'])
        utt2spk=args.train_utt2spk,
        spk2gender=args.train_spk2gender,
        perturber=perturber
    )
    if rank == 0:
        logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = AudioMelSCPDataset(
        wav_scp=args.dev_wav_scp,
        prompt_scp=args.dev_prompt_scp,
        target_scp=args.dev_target_scp,
        aux_scp=args.dev_aux_scp,
        idx_scp=args.dev_idx_scp,
        utt2num_frames=args.dev_num_frames,
        segments=args.dev_segments,
        min_num_frames=config.get("min_num_frames", None),
        max_num_frames=config.get("max_num_frames", None),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
        length_tolerance=config.get("length_tolerance", 2),
        prompt_min_length=config.get("prompt_min_length", 500),
        extractor_fs=config['feature_extractor_params'].get("fs", 16000),
        # speed_alpha_range=eval(config['speed_alpha_range'])
        utt2spk=args.dev_utt2spk,
        spk2gender=args.dev_spk2gender,
        perturber=perturber
    )
    if rank == 0:
        logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collator = Collator(
        hop_size_for_extractor=config['feature_extractor_params']['hop_size'],
        win_length_for_extractor=config['feature_extractor_params']['win_length'],
        sampling_rate=config["feature_extractor_params"]["fs"],
        n_mel=1024,
        force_from_start=config.get("force_from_start", False)
    )

    sampler = {
        "train": DistributedSampler(
            dataset=dataset["train"],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        ),
        "dev": DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=False,
        )}
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False,
            collate_fn=collator,
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False,
            collate_fn=collator,
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    from lscodec.utils import load_model
    config['pretrain_codebook'] = args.pretrain_codebook
    generator = load_model(config, None)

    model = {
        "generator": generator.to(device),
    }

    # define optimizers and schedulers
    generator_optimizer_class = getattr(
        lscodec.optimizers,
        # keep compatibility
        config.get("generator_optimizer_type", "RAdam"),
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("generator_scheduler_type", "StepLR"),
    )
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
    }
    from torch.nn.parallel import DistributedDataParallel
    model["generator"] = DistributedDataParallel(model["generator"], device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        # show settings
        logging.info(model["generator"])
        logging.info(optimizer["generator"])

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    if len(args.pretrain_vae) != 0:
        trainer.load_vae_params(args.pretrain_vae)
        if rank == 0:
            logging.info(f"Successfully load parameters from {args.pretrain_vae}.")

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        if rank == 0:
            logging.info(f"Successfully load parameters from {args.pretrain}.")

    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        if rank == 0:
            logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    finally:
        if rank == 0:
            trainer.save_checkpoint(
                os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    print(f"============> using {n_gpus} GPUS")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8001"

    mp.spawn(
        main,
        nprocs=n_gpus,
        args=(n_gpus,)
    )
