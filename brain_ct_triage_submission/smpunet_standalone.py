# -*- coding: utf-8 -*-
"""
smpunet_standalone.py — پیاده‌سازی pure-PyTorch معماری U-Net با انکودر ResNet34
(معادل `smp.Unet(encoder_name="resnet34", in_channels=1, classes=3)` در
segmentation-models-pytorch 0.5.x)

این فایل فقط یک «پیاده‌سازی بدون وابستگی» است تا بسته‌ی سابمیت بدون نیاز به
کتابخانه‌ی segmentation-models-pytorch در محیط داوری اجرا شود.

ساختار و نام کلیدها key-to-key با چک‌پوینت‌های تیم میدلاین (`cv_fold*_best.zip`
با پیشوند `model.`) سازگار است و تساوی عددی خروجی آن با U-Net خودِ smp
به‌صورت محلی تأیید شده است.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34 as _tv_resnet34

DECODER_CHANNELS = (256, 128, 64, 32, 16)
ENCODER_CHANNELS = (3, 64, 64, 128, 256, 512)  # resnet34 (اولی ورودی RGB؛ استفاده نمی‌شود)


class Conv2dReLU(nn.Sequential):
    """معادل md.Conv2dReLU با use_norm="batchnorm" (Conv بدون bias + BN + ReLU)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = Conv2dReLU(in_ch + skip_ch, out_ch)
        self.attention1 = nn.Identity()
        self.conv2 = Conv2dReLU(out_ch, out_ch)
        self.attention2 = nn.Identity()

    def forward(self, feature_map, target_h, target_w, skip_connection=None):
        feature_map = F.interpolate(feature_map, size=(target_h, target_w), mode="nearest")
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
            feature_map = self.attention1(feature_map)
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.attention2(feature_map)
        return feature_map


class ResNet34Encoder(nn.Module):
    """معادل SMP ResNetEncoder برای resnet34 — خروجی ۶ نقشه‌ی ویژگی."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        net = _tv_resnet34(weights=None)
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = (in_channels,) + ENCODER_CHANNELS[1:]

    def forward(self, x):
        # SMP ResNetEncoder خودِ ورودی را هم به‌عنوان feature[0] برمی‌گرداند
        # تا دکودر در انتها به رزولوشن کامل برسد.
        s0 = self.relu(self.bn1(self.conv1(x)))   # 64,  H/2
        s1 = self.layer1(self.maxpool(s0))         # 64,  H/4
        s2 = self.layer2(s1)                       # 128, H/8
        s3 = self.layer3(s2)                       # 256, H/16
        s4 = self.layer4(s3)                       # 512, H/32
        return [x, s0, s1, s2, s3, s4]


class UnetDecoder(nn.Module):
    def __init__(self,
                 encoder_channels=ENCODER_CHANNELS,
                 decoder_channels=DECODER_CHANNELS):
        super().__init__()
        enc = list(encoder_channels[1:])[::-1]      # (512, 256, 128, 64, 64)
        head = enc[0]
        in_channels = [head] + list(decoder_channels[:-1])
        skip_channels = enc[1:] + [0]
        self.center = nn.Identity()
        self.blocks = nn.ModuleList([
            UnetDecoderBlock(i, s, o)
            for i, s, o in zip(in_channels, skip_channels, decoder_channels)
        ])

    def forward(self, features):
        spatial_shapes = [f.shape[2:] for f in features][::-1]
        feats = features[1:][::-1]
        x = self.center(feats[0])
        skips = feats[1:]
        for i, block in enumerate(self.blocks):
            h, w = spatial_shapes[i + 1]
            skip = skips[i] if i < len(skips) else None
            x = block(x, h, w, skip_connection=skip)
        return x


class SMPUnetResNet34(nn.Module):
    """معادل `smp.Unet(encoder_name="resnet34", in_channels=1, classes=3)`.

    خروجی logits با همان ساختار key naming درونی (`encoder.*`, `decoder.*`,
    `segmentation_head.*`) تا با state_dict چک‌پوینت‌ها strict لود شود.
    """

    def __init__(self, in_channels: int = 1, classes: int = 3):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels)
        self.decoder = UnetDecoder()
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(DECODER_CHANNELS[-1], classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        return self.segmentation_head(decoder_output)


class MidlineHeatmapModel(nn.Module):
    """Wrapper هم‌نام با کلاس ترین تیم میدلاین (state_dict با پیشوند `model.`)."""

    def __init__(self, in_channels: int = 1, classes: int = 3):
        super().__init__()
        self.model = SMPUnetResNet34(in_channels, classes)

    def forward(self, x):
        return self.model(x)
