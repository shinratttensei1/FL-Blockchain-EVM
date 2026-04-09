"""SE-ResNet model and loss functions for MHEALTH activity classification.

Architecture: 4-stage SE-ResNet with squeeze-and-excitation blocks.
Input: (B, 23, 256) -- 23 sensor channels, 256-sample sliding windows.
Output: 12 activity class logits.

Adapted from the original ECG SE-ResNet (Jimenez et al., arXiv:2208.10993v3)
for mobile health human activity recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fl_blockchain_evm.core.constants import NUM_CLASSES, NUM_CHANNELS


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        pt = torch.sigmoid(logits) * targets + \
            (1 - torch.sigmoid(logits)) * (1 - targets)
        focal = (1 - pt).clamp(min=1e-6) ** self.gamma * bce
        if self.alpha is not None:
            focal = (self.alpha / self.alpha.mean()).unsqueeze(0) * focal
        return focal.mean()


class _SEResBlock(nn.Module):
    def __init__(self, ch, ks=5):
        super().__init__()
        pad = ks // 2
        self.conv1 = nn.Conv1d(ch, ch, ks, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, ks, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        mid = max(ch // 4, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out * self.se(out).unsqueeze(-1)
        return F.relu(out + x, inplace=True)


class Net(nn.Module):
    """4-stage SE-ResNet: (B, 23, 256) -> 12 activity logits.

    Pooling schedule (stride-2 at each stage) keeps the temporal
    dimension manageable for 256-sample windows:
      256 -> 128 -> 64 -> 32 -> 16 -> GAP -> 256-d embedding -> 12 logits
    """

    def __init__(self, num_classes=NUM_CLASSES, in_channels=NUM_CHANNELS):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)

        # Stage 1: (B, 23, 256) -> pool -> (B, 32, 128) -> SE blocks
        self.conv1 = nn.Conv1d(in_channels, 32, 7, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.res1a = _SEResBlock(32, 5)
        self.res1b = _SEResBlock(32, 5)

        # Stage 2: (B, 32, 128) -> pool -> (B, 64, 64) -> SE blocks
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.res2a = _SEResBlock(64, 3)
        self.res2b = _SEResBlock(64, 3)

        # Stage 3: (B, 64, 64) -> pool -> (B, 128, 32) -> SE blocks
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.res3a = _SEResBlock(128, 3)
        self.res3b = _SEResBlock(128, 3)

        # Stage 4: (B, 128, 32) -> pool -> (B, 256, 16) -> SE block
        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, bias=False)
        self.bn4   = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)
        self.res4  = _SEResBlock(256, 3)

        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.res1b(self.res1a(self.pool1(
            F.relu(self.bn1(self.conv1(x)), inplace=True))))
        x = self.res2b(self.res2a(self.pool2(
            F.relu(self.bn2(self.conv2(x)), inplace=True))))
        x = self.res3b(self.res3a(self.pool3(
            F.relu(self.bn3(self.conv3(x)), inplace=True))))
        x = self.res4(self.pool4(
            F.relu(self.bn4(self.conv4(x)), inplace=True)))
        return self.fc(self.drop(self.gap(x).squeeze(-1)))
