# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2025 Yiwei Guo

import torch
from typing import Union

from lscodec.utils import crop_seq
from lscodec.models.quantization.core_vq import GroupVectorQuantization
from lscodec.models.fairseq_modules.fp32_group_norm import Fp32GroupNorm
from lscodec.models.feature_extractor import LSCodecEncoder
from lscodec.models.prompt_prenet import ConvPromptPrenet
import logging


class QuantizerProjection(torch.nn.Module):
    def __init__(self, dim, groups):
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False),
            Fp32GroupNorm(groups, dim),
        )

    def forward(self, x):
        return self.projection(x)


class LSCodecEncoderQuantizer(torch.nn.Module):

    def __init__(self, frontend: LSCodecEncoder,
                 quantizer: GroupVectorQuantization,
                 dropout_features: float,
                 mean_only: bool=False,
                 length_mismatch_tolerance: int = 3):
        super(LSCodecEncoderQuantizer, self).__init__()
        self.frontend = frontend
        self.quantizer = quantizer

        self.dropout_feats = torch.nn.Dropout(p=dropout_features)
        self.length_mismatch_tolerance = length_mismatch_tolerance
        self.mean_only = mean_only

    def sample_from_posterior(self, means, logvars):
        # z = mean + N(0,1)*exp(0.5*logvars)
        return means + torch.randn_like(means) * torch.exp(0.5*logvars)

    def forward(self, waveform: torch.Tensor, prompt, mask=None, prompt_mask=None, perform_sample=True):
        features = self.frontend.forward(waveform, mask, self.length_mismatch_tolerance)  # [B, D, L]
        B, D, L = features.shape
        if mask is not None:
            features = features.masked_fill(~mask.unsqueeze(1), 0)
        features = features.transpose(1, 2)  # [B, L, D]

        # get mean and logvars from features
        means = features[..., :D//2]
        
        if self.mean_only:
            logvars = torch.zeros_like(means)
        else:
            logvars = features[..., D//2:]

        if self.quantizer is not None:
            if mask is not None:
                features, embed_index, commitment_loss = self.quantizer.forward(means[mask])  # NOTE: quantize on mean only.
                # NOTE: the input is already a mask-selected tensor [something, D]
                # now features is a [something, D] tensor. We need to convert it back
                L = mask.shape[-1]
                vqvec = torch.zeros(B, L, D//2).float().to(features.device)
                vqvec[mask.unsqueeze(-1).expand(-1, -1, D//2)] = features.view(-1)
            else:
                features, embed_index, commitment_loss = self.quantizer.forward(means.reshape(B*L, D//2))
                vqvec = features.reshape(B, L, D//2)
            vqvec = vqvec.transpose(1, 2)  # still [B, D, L] for compatibility.

            vqvec = self.dropout_feats(vqvec)

            if mask is not None and vqvec.shape[-1] != mask.shape[-1]:
                if abs(vqvec.shape[-1] - mask.shape[-1]) <= self.length_mismatch_tolerance:
                    min_len = min(vqvec.shape[-1], mask.shape[-1])
                    vqvec = vqvec[..., :min_len]
                    mask = mask[..., :min_len]
                else:
                    raise RuntimeError(f"After feature extractor, length mismatch: "
                                       f"extracted {vqvec.shape[-1]} frames and original mel is {mask.shape[-1]} frames")
        else:
            if perform_sample:
                features = self.sample_from_posterior(means, logvars)
            else:
                features = means
            vqvec = features.transpose(1, 2)  # NOTE: a little dirty hack. This is not actually "VQ" vector.
            commitment_loss = None

        return commitment_loss, mask, means, logvars

    def encode(self, waveform: torch.Tensor, mask=None):

        features = self.frontend.forward(waveform, mask, self.length_mismatch_tolerance)  # [B, D, L]
        B, D, L = features.shape
        features = features.transpose(1, 2)  # [B, L, D]

        # get mean and logvars from features
        means = features[..., :D//2]
        logvars = features[..., D//2:]
        if self.quantizer is not None:
            features, embed_index, commitment_loss = self.quantizer.forward(means.reshape(B*L, D//2))
        else:
            embed_index = None
        if embed_index is not None and len(embed_index.shape) == 1:
            embed_index = embed_index.unsqueeze(-1)  # [L, 1]
        return means, logvars, embed_index
