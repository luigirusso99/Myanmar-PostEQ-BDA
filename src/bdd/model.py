from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet18


class BuildingGuidedFusion(nn.Module):
    """
    Building-guided multimodal fusion for building damage detection.

    The footprint feature acts as a building-location prior. SAR and RGB features are
    fused with footprint-conditioned gates and then refined by a local spatial
    attention map. This is lighter and more stable than full cross-attention for
    small building-centered patches.
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.sar_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.rgb_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        ftp_feat: torch.Tensor,
        sar_feat: torch.Tensor | None = None,
        rgb_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sar_feat is None and rgb_feat is None:
            raise ValueError("BuildingGuidedFusion requires at least SAR or RGB features.")

        if sar_feat is None:
            sar_feat = torch.zeros_like(ftp_feat)
        if rgb_feat is None:
            rgb_feat = torch.zeros_like(ftp_feat)

        sar_gate = self.sar_gate(torch.cat([ftp_feat, sar_feat], dim=1))
        rgb_gate = self.rgb_gate(torch.cat([ftp_feat, rgb_feat], dim=1))

        sar_guided = sar_feat * sar_gate
        rgb_guided = rgb_feat * rgb_gate

        fused = self.fuse(torch.cat([ftp_feat, sar_guided, rgb_guided], dim=1))
        attention = self.spatial_attention(fused)

        return ftp_feat + fused * attention


def make_resnet18_trunk_and_tail(weights_init=None):
    backbone = resnet18(weights=weights_init)
    trunk = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    )
    tail = backbone.layer4
    return trunk, tail

class MultiModalBuildingGuidedFusion(nn.Module):
    """Mid-fusion network with building-guided gated fusion over SAR/RGB features."""

    def __init__(
        self,
        embed_dim: int = 128,
        use_sar: bool = True,
        use_rgb: bool = True,
        weights_init=None,
    ):
        super().__init__()
        self.use_sar = use_sar
        self.use_rgb = use_rgb

        if not self.use_sar and not self.use_rgb:
            raise ValueError("MultiModalBuildingGuidedFusion requires at least one of use_sar or use_rgb to be True.")

        self.ftp_trunk, self.ftp_tail = make_resnet18_trunk_and_tail(weights_init)

        if self.use_sar:
            self.sar_trunk, _ = make_resnet18_trunk_and_tail(weights_init)

        if self.use_rgb:
            self.rgb_trunk, _ = make_resnet18_trunk_and_tail(weights_init)

        self.fusion = BuildingGuidedFusion(dim=256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        ftp: torch.Tensor,
        sar: torch.Tensor | None = None,
        rgb: torch.Tensor | None = None,
    ):
        ftp_feat = self.ftp_trunk(ftp)
        sar_feat = None
        rgb_feat = None

        if self.use_sar:
            if sar is None:
                raise ValueError("SAR branch is active, but sar=None.")
            sar_feat = self.sar_trunk(sar)

        if self.use_rgb:
            if rgb is None:
                raise ValueError("RGB branch is active, but rgb=None.")
            rgb_feat = self.rgb_trunk(rgb)

        fused_feat = self.fusion(ftp_feat, sar_feat=sar_feat, rgb_feat=rgb_feat)
        out_feat = self.ftp_tail(fused_feat)
        pooled = self.pool(out_feat).flatten(1)
        return self.classifier(pooled).squeeze(1)


if __name__ == "__main__":
    batch_size = 4
    height = 224
    width = 224

    ftp = torch.randn(batch_size, 3, height, width)
    sar = torch.randn(batch_size, 3, height, width)
    rgb = torch.randn(batch_size, 3, height, width)

    model = MultiModalBuildingGuidedFusion(use_sar=True, use_rgb=True)
    logits = model(ftp=ftp, sar=sar, rgb=rgb)
    print("Building-guided fusion output shape:", logits.shape)
