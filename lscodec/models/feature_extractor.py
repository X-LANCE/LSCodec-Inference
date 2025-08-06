import math

import torch.nn as nn
from lscodec.models.fairseq_modules.fp32_group_norm import Fp32GroupNorm
from lscodec.models.fairseq_modules.layer_norm import Fp32LayerNorm
from lscodec.models.fairseq_modules.transpose_last import TransposeLast
import torch


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod


class ZeroPad1d(nn.Module):
    def __init__(self, pad_left, pad_right):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        return nn.functional.pad(x, (self.pad_left, self.pad_right))


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers,
            dropout,
            log_compression,
            skip_connections,
            residual_scale,
            non_affine_group_norm,
            activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm
                ),
                activation,
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):  # x is audio of shape [B, 1, T]
        # Bx1xT -> BxCxT
        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        return x


class ConvAggregator(nn.Module):
    def __init__(
        self,
        conv_layers,
        embed,
        dropout,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        conv_bias,
        zero_pad,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            # padding dims only really make sense for stride = 1
            ka = k // 2
            kb = ka - 1 if k % 2 == 0 else ka

            pad = (
                ZeroPad1d(ka + kb, 0) if zero_pad else nn.ReplicationPad1d((ka + kb, 0))
            )

            return nn.Sequential(
                pad,
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias),
                nn.Dropout(p=dropout),
                norm_block(False, n_out, affine=not non_affine_group_norm),
                activation,
            )

        in_d = embed
        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        for dim, k, stride in conv_layers:
            if in_d != dim and skip_connections:
                self.residual_proj.append(nn.Conv1d(in_d, dim, 1, bias=False))
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = conv(x)
            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                x = (x + residual) * self.residual_scale
        return x


class LSCodecEncoder(nn.Module):
    def __init__(self, conv_extraction_model: ConvFeatureExtractionModel,
                 conv_aggregation_model: ConvAggregator,
                 dropout_p: float = 0.1):
        super().__init__()
        self.conv_models = nn.Sequential(conv_extraction_model, conv_aggregation_model)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, waveform, frame_mask, length_tolerance=3):
        # waveform: [B, 1, T]. Frame mask is 1 at valid values.
        conv_features = self.conv_models(waveform)  # [B, 1024, L]
        conv_features = self.dropout(conv_features)
        B, D, L = conv_features.shape
        if frame_mask is None:
            frame_mask = torch.ones(size=(B, L)).bool().to(conv_features.device)
        if (frame_mask is not None) and (0 < frame_mask.shape[-1] - L <= length_tolerance):
            pad_tensor = torch.zeros(size=(B, D, frame_mask.shape[-1] - L)).to(conv_features.device)
            # NOTE: we don't pad these values with 999, but instead treat those as valid data points
            # NOTE: this is because the mask is longer than the valid data points, and changing the mask is inconvenient.
            conv_features = torch.concat([conv_features, pad_tensor], dim=-1)

        return conv_features
