# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Layer norm done in fp32 (for fp16 training)
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

from math import ceil, floor

import torch


def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        # print("In FP32 GroupNorm", input[0].mean(), input[0].std(), input[0].min(), input[0].max())
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        
        # NOTE: dirty hack
        # if input.max()<1000:
        #     output = F.group_norm(
        #             input.float(),
        #             self.num_groups,
        #             self.weight.float() if self.weight is not None else None,
        #             self.bias.float() if self.bias is not None else None,
        #             self.eps,
        #         )
        # else:
        #     print("Will remove outliers in a dirty way in FP32GroupNorm")
        #     q1 = torch_quantile(input.view(-1), 0.25)
        #     q3 = torch_quantile(input.view(-1), 0.75)
        #     IQR = q3-q1
        #     upper_bound = q3 + 1.5 * IQR
        #     lower_bound = q1 - 1.5 * IQR
        #     # correct_mean = input[(lower_bound<input) & (input<upper_bound)].mean()
        #     # correct_std = input[(lower_bound<input) & (input<upper_bound)].std()
        #     # print(f"After removing outliers, the correct mean and std are {correct_mean}, {correct_std}")
        #     # output = (input-correct_mean)/(correct_std+self.eps) * self.weight.float().unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)
            
        #     input[(lower_bound<input) & (input<upper_bound)] = 0.0
        #     output = F.group_norm(
        #             input.float(),
        #             self.num_groups,
        #             self.weight.float() if self.weight is not None else None,
        #             self.bias.float() if self.bias is not None else None,
        #             self.eps,
        #         )
            
        return output.type_as(input)
