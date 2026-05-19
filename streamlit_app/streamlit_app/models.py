"""
Model definitions for 3 UNet+ASPP variants.

- OLD  : ResNet34UNetASPP       — binary sigmoid, input 9ch (RGB + coarse + co_occ + one_hot4)
- v2   : ResNet34UNetASPPMulti  — softmax 5ch, input 8ch (RGB + 4 coarses + co_occ)
- v3   : ResNet34UNetASPPMultiV3— softmax 5ch, input 8ch + DINOv2 feature (384 → 32 compressed)

Kiến trúc chung: ResNet-34 encoder (pretrained ImageNet) + ASPP bottleneck (dilation 1,6,12,18,
GAP branch) + UNet decoder với skip connections.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLab-V3).

    - 4 nhánh: 1x1 conv + 3 atrous conv (rate 6, 12, 18) — capture multi-scale context.
    - 1 nhánh GAP: global average pool -> 1x1 conv -> upsample -> global context.
      Dùng GroupNorm thay BatchNorm vì spatial=(1,1) khi GAP; BN sẽ NaN khi batch=1 eval.
    - Project: concat 5 nhánh (5×out_ch channel) -> 1x1 conv về out_ch, có Dropout2d(0.1).
    """

    def __init__(self, in_ch: int = 512, out_ch: int = 256) -> None:
        super().__init__()

        def _branch(dilation: int) -> nn.Sequential:
            if dilation == 1:
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.b1 = _branch(1)
        self.b6 = _branch(6)
        self.b12 = _branch(12)
        self.b18 = _branch(18)

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        gap = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.project(
            torch.cat([self.b1(x), self.b6(x), self.b12(x), self.b18(x), gap], dim=1)
        )


class DecoderBlock(nn.Module):
    """UNet decoder block: transposed conv upsample + concat skip + 2x (3x3 conv + BN + ReLU)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


def _make_resnet34_encoder(in_channels: int):
    """ResNet-34 backbone with first conv expanded to `in_channels`.

    - Copy RGB weight (3ch) từ ImageNet pretrained.
    - Init extra channels = 0 để giữ RGB forward giống pretrain.
    """
    backbone = tvm.resnet34(weights=None)

    orig_w = backbone.conv1.weight.data.clone()  # [64, 3, 7, 7]
    new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = orig_w
        new_conv.weight[:, 3:] = 0.0
    backbone.conv1 = new_conv
    return backbone


class ResNet34UNetASPP(nn.Module):
    """OLD variant — binary sigmoid, input = 9 channels.

    Input layout: [RGB(3), coarse(1), co_occ(1), one_hot(4)]
    Output: [B, 1, H, W] logits -> sigmoid -> prob of "is target class".
    Inference: forward 4 lần/ảnh (mỗi lần với one-hot class khác) -> argmax qua 4 prob map.
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 1) -> None:
        super().__init__()
        backbone = _make_resnet34_encoder(in_channels)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.aspp = ASPP(in_ch=512, out_ch=256)

        self.dec4 = DecoderBlock(256, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        ep = self.pool(e0)
        e1 = self.enc1(ep)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottle = self.aspp(e4)
        d = self.dec4(bottle, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, e0)
        d = self.final_up(d)
        return self.head(d)


class ResNet34UNetASPPMulti(nn.Module):
    """v2 variant — multi-class softmax, input = 8 channels.

    Input layout: [RGB(3), coarse_table(1), coarse_sofa(1), coarse_chair(1), coarse_bike(1), co_occ(1)]
    Output: [B, 5, H, W] logits -> softmax -> argmax ∈ {0..4}
      0=bg, 1=diningtable, 2=sofa, 3=chair, 4=bicycle.
    """

    def __init__(self, in_channels: int = 8, out_channels: int = 5) -> None:
        super().__init__()
        backbone = _make_resnet34_encoder(in_channels)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.aspp = ASPP(in_ch=512, out_ch=256)
        self.dec4 = DecoderBlock(256, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        ep = self.pool(e0)
        e1 = self.enc1(ep)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottle = self.aspp(e4)
        d = self.dec4(bottle, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, e0)
        d = self.final_up(d)
        return self.head(d)


class ResNet34UNetASPPMultiV3(nn.Module):
    """v3 variant — multi-class softmax + DINOv2.

    Forward args:
      - input_8ch  : [B, 8, H, W]   — RGB + 4 coarses + co_occ
      - dino_feat  : [B, 384, 37, 37] — raw DINOv2 feature (patch tokens reshape)
    Pipeline:
      dino_feat -> 1x1 conv compress to `dinov2_compress` (32) -> bilinear upsample to (H, W)
      -> concat với input_8ch -> total `8 + 32 = 40` channels vào ResNet.
    """

    def __init__(
        self,
        in_channels_8: int = 8,
        dinov2_dim: int = 384,
        dinov2_compress: int = 32,
        out_channels: int = 5,
    ) -> None:
        super().__init__()
        self.in_channels_8 = in_channels_8
        self.dinov2_dim = dinov2_dim
        self.dinov2_compress = dinov2_compress
        self.total_in = in_channels_8 + dinov2_compress

        self.dino_compress = nn.Sequential(
            nn.Conv2d(dinov2_dim, dinov2_compress, 1, bias=False),
            nn.BatchNorm2d(dinov2_compress),
            nn.ReLU(inplace=True),
        )

        backbone = _make_resnet34_encoder(self.total_in)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4

        self.aspp = ASPP(in_ch=512, out_ch=256)
        self.dec4 = DecoderBlock(256, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, input_8ch: torch.Tensor, dino_feat: torch.Tensor) -> torch.Tensor:
        H, W = input_8ch.shape[-2:]
        dino_c = self.dino_compress(dino_feat)
        dino_u = F.interpolate(dino_c, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([input_8ch, dino_u], dim=1)

        e0 = self.enc0(x)
        ep = self.pool(e0)
        e1 = self.enc1(ep)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottle = self.aspp(e4)
        d = self.dec4(bottle, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, e0)
        d = self.final_up(d)
        return self.head(d)
