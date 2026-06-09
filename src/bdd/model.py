import torch
import torch.nn as nn
from torchvision.models import resnet18


class MultiModalSARFTPRGB(nn.Module):
    """Late-fusion network: SAR branch + footprint branch + optional RGB branch."""

    def __init__(self, embed_dim: int = 128, use_sar: bool = True, use_rgb: bool = True, weights_init=None):
        super().__init__()
        self.use_sar = use_sar
        self.use_rgb = use_rgb

        if self.use_sar:
            self.branch_sar = resnet18(weights=weights_init)
            self.branch_sar.fc = nn.Linear(self.branch_sar.fc.in_features, embed_dim)

        self.branch_ftp = resnet18(weights=weights_init)
        self.branch_ftp.fc = nn.Linear(self.branch_ftp.fc.in_features, embed_dim)

        if self.use_rgb:
            self.branch_rgb = resnet18(weights=weights_init)
            self.branch_rgb.fc = nn.Linear(self.branch_rgb.fc.in_features, embed_dim)

        n_branches = 1 + int(use_sar) + int(use_rgb)
        self.classifier = nn.Sequential(
            nn.Linear(n_branches * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, ftp: torch.Tensor, sar: torch.Tensor | None = None, rgb: torch.Tensor | None = None):
        feats = [self.branch_ftp(ftp)]

        if self.use_sar:
            if sar is None:
                raise ValueError("SAR branch is active, but sar=None.")
            feats.append(self.branch_sar(sar))

        if self.use_rgb:
            if rgb is None:
                raise ValueError("RGB branch is active, but rgb=None.")
            feats.append(self.branch_rgb(rgb))

        return self.classifier(torch.cat(feats, dim=1)).squeeze(1)
