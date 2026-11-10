"""Dual-stream point cloud segmentation network.

Two encoders run on the same xyz input and their per-point features are
concatenated before a shared segmentation head:

* ``CNN_residual`` - a per-point residual MLP (1x1 convolutions).
* ``dgcnn``        - a DGCNN encoder (EdgeConv on a dynamic k-NN graph).

Attribute names deliberately match the original training code so that the
checkpoints released with the paper load without any key renaming.
"""

import torch
import torch.nn as nn


def knn(x, k):
    """Indices of the k nearest neighbours of every point. x: (B, C, N)."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)


def get_graph_feature(x, k):
    """EdgeConv input [x_j - x_i, x_i] for the k neighbours j of each point i.

    x: (B, C, N) -> (B, 2C, N, k)
    """
    batch_size, num_dims, num_points = x.size()
    idx = knn(x, k=k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()


class ResidualBlock(nn.Module):
    """Two 1x1 conv + BN layers with a projected skip connection."""

    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return self.activation(out)


class CNNResidualStream(nn.Module):
    """Per-point residual encoder. (B, 3, N) -> (B, 512, N)."""

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.block1 = ResidualBlock(3, 64, self.activation)
        self.block2 = ResidualBlock(64, 128, self.activation)
        self.block3 = ResidualBlock(128, 256, self.activation)
        self.block4 = ResidualBlock(256, 512, self.activation)
        self.conv5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   self.activation)
        self.conv6 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   self.activation)
        self.conv7 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   self.activation)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, 3, N, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv5(x).squeeze(-1)
        x = self.conv6(x)
        return self.conv7(x)


class DGCNNStream(nn.Module):
    """DGCNN encoder with three EdgeConv layers. (B, 3, N) -> (B, 512, N)."""

    def __init__(self, k=40):
        super().__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)  # unused; kept so released checkpoints load strictly

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5, nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6, nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7, nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        num_points = x.size(2)

        x = self.conv2(self.conv1(get_graph_feature(x, self.k)))
        x1 = x.max(dim=-1)[0]                                   # (B, 64, N)
        x = self.conv4(self.conv3(get_graph_feature(x1, self.k)))
        x2 = x.max(dim=-1)[0]                                   # (B, 64, N)
        x = self.conv5(get_graph_feature(x2, self.k))
        x3 = x.max(dim=-1)[0]                                   # (B, 64, N)

        x = torch.cat((x1, x2, x3), dim=1)                      # (B, 192, N)
        x = self.conv6(x).max(dim=-1, keepdim=True)[0]          # global feature (B, 1024, 1)
        x = torch.cat((x.repeat(1, 1, num_points), x1, x2, x3), dim=1)
        return self.conv7(x)                                    # (B, 512, N)


class DualStreamNetwork(nn.Module):
    """(B, 3, N) xyz -> (B, num_classes, N) logits."""

    def __init__(self, num_classes=8, k=40, dropout=0.35):
        super().__init__()
        self.CNN_residual = CNNResidualStream()
        self.dgcnn = DGCNNStream(k=k)
        self.conv8 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Conv1d(512, num_classes, kernel_size=1, bias=False)

    def forward(self, points):
        x = torch.cat((self.CNN_residual(points), self.dgcnn(points)), dim=1)
        x = self.dp1(self.conv8(x))
        return self.conv9(x)

    def reset_head(self, num_classes):
        """Replace the last layer, e.g. before fine-tuning on a new label set."""
        self.conv9 = nn.Conv1d(512, num_classes, kernel_size=1, bias=False).to(
            self.conv8[0].weight.device)

    # Trained during fine-tuning (everything else is frozen), and the output layer.
    HEAD = ("conv8", "conv9")
    LAST = "conv9"
