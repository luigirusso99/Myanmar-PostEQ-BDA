from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet18


class FGCA(nn.Module):
    """Footprint-guided cross attention over SAR/RGB feature maps."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, ftp_feat: torch.Tensor, context_feats: list[torch.Tensor]) -> torch.Tensor:
        if len(context_feats) == 0:
            raise ValueError("FGCA requires at least one context feature map.")

        b, c, h, w = ftp_feat.shape
        query = ftp_feat.flatten(2).transpose(1, 2)
        key_value = torch.cat(
            [feat.flatten(2).transpose(1, 2) for feat in context_feats],
            dim=1,
        )
        fused, _ = self.attn(query=query, key=key_value, value=key_value)
        return fused.transpose(1, 2).reshape(b, c, h, w)


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

class MultiModalFGCA(nn.Module):
    """Mid-fusion network with footprint-guided cross attention over SAR/RGB features."""

    def __init__(
        self,
        embed_dim: int = 128,
        use_sar: bool = True,
        use_rgb: bool = True,
        weights_init=None,
        num_heads: int = 4,
    ):
        super().__init__()
        self.use_sar = use_sar
        self.use_rgb = use_rgb

        if not self.use_sar and not self.use_rgb:
            raise ValueError("MultiModalFGCA requires at least one of use_sar or use_rgb to be True.")

        self.ftp_trunk, self.ftp_tail = make_resnet18_trunk_and_tail(weights_init)

        if self.use_sar:
            self.sar_trunk, _ = make_resnet18_trunk_and_tail(weights_init)

        if self.use_rgb:
            self.rgb_trunk, _ = make_resnet18_trunk_and_tail(weights_init)

        self.fgca = FGCA(dim=256, num_heads=num_heads)
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
        context_feats = []

        if self.use_sar:
            if sar is None:
                raise ValueError("SAR branch is active, but sar=None.")
            context_feats.append(self.sar_trunk(sar))

        if self.use_rgb:
            if rgb is None:
                raise ValueError("RGB branch is active, but rgb=None.")
            context_feats.append(self.rgb_trunk(rgb))

        fused_feat = self.fgca(ftp_feat, context_feats)
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

    fgca_model = MultiModalFGCA(use_sar=True, use_rgb=True)
    fgca_logits = fgca_model(ftp=ftp, sar=sar, rgb=rgb)
    print("MultiModalFGCA output shape:", fgca_logits.shape)
