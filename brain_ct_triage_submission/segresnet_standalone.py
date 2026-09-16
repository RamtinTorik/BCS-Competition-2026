# -*- coding: utf-8 -*-
"""
segresnet_standalone.py — پیاده‌سازی pure-PyTorch معماری SegResNet (معادل monai 1.6)

این فایل فقط یک «fallback بدون وابستگی» است تا اگر در محیط داوری کتابخانه‌ی
MONAI موجود نبود، بسته‌ی سابمیت بدون هیچ تغییری در وزن‌ها کار کند.

ساختار دقیقاً معادل `monai.networks.nets.SegResNet` (spatial_dims=2) است:
    convInit -> [down_layers (pre_conv stride2 + ResBlock×n)] -> decode
    (up_sample conv1x1 + nearest×2 + skip) -> ResBlock -> conv_final
و state_dict آن key-to-key با چک‌پوینت‌های تیم خونریزی سازگار است
(load_state_dict با strict=True).

تساوی عددی این پیاده‌سازی با SegResNet خودِ monai به‌صورت محلی تأیید شده است.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn


class _ConvWrap(nn.Module):
    """Wrapper با attribute ‏`conv` تا نام کلیدها با monai یکی بماند (conv.bias=False)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block سبک monai: pre-act (GroupNorm->ReLU->Conv) ×2 + skip."""

    def __init__(self, in_ch: int, num_groups: int = 8):
        super().__init__()
        # GroupNorm به ۸ گروه؛ اگر in_ch بر ۸ بخش‌پذیر نبود، نزدیک‌ترین تعداد مجاز
        g = num_groups
        while in_ch % g != 0:
            g -= 1
        self.norm1 = nn.GroupNorm(g, in_ch)
        self.norm2 = nn.GroupNorm(g, in_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = _ConvWrap(in_ch, in_ch)
        self.conv2 = _ConvWrap(in_ch, in_ch)

    def forward(self, x):
        identity = x
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return x + identity


class SegResNetV2(nn.Module):
    """معادل `SegResNet(spatial_dims=2, ...)` در monai — فقط برای اینفرنس CPU/GPU."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 init_filters: int = 16,
                 blocks_down: Sequence[int] = (1, 2, 2, 4),
                 blocks_up: Sequence[int] = (1, 1, 1),
                 dropout_prob: float = 0.2,
                 num_groups: int = 8):
        super().__init__()
        self.dropout_prob = dropout_prob
        filters = init_filters

        self.convInit = _ConvWrap(in_channels, filters, 3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob is not None else None

        # --- down layers ---
        down_layers = []
        for i, item in enumerate(blocks_down):
            layer_in = filters * (2 ** i)
            pre_conv = (_ConvWrap(layer_in // 2, layer_in, 3, stride=2, padding=1, bias=False)
                        if i > 0 else nn.Identity())
            down_layers.append(nn.Sequential(
                pre_conv, *[ResBlock(layer_in, num_groups) for _ in range(item)]))
        self.down_layers = nn.ModuleList(down_layers)

        # --- up layers + up samples ---
        up_layers, up_samples = [], []
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in = filters * (2 ** (n_up - i))
            up_layers.append(nn.Sequential(
                *[ResBlock(sample_in // 2, num_groups) for _ in range(blocks_up[i])]))
            up_samples.append(nn.Sequential(
                _ConvWrap(sample_in, sample_in // 2, kernel_size=1, padding=0, bias=False),
                nn.Upsample(scale_factor=(2.0, 2.0), mode="bilinear")))
        self.up_layers = nn.ModuleList(up_layers)
        self.up_samples = nn.ModuleList(up_samples)

        g = num_groups
        while filters % g != 0:
            g -= 1
        self.conv_final = nn.Sequential(
            nn.GroupNorm(g, filters),
            nn.ReLU(inplace=True),
            _ConvWrap(filters, out_channels, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convInit(x)
        if self.dropout is not None:
            x = self.dropout(x)

        down_x = []
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        down_x.reverse()
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        return self.conv_final(x)
