"""Single-stream baselines of the paper (Table 1): PointNet, DGCNN, CNN-Residual.

Layer names follow the original training code (Model.py / Model2.py), so its
checkpoints load. The networks return logits; the cross-entropy loss applies
the softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dualstream import ResidualBlock, get_graph_feature


# --------------------------------------------------------------------------
# PointNet (segmentation version, PointNetNew1 in the original code)
# --------------------------------------------------------------------------

class STN3d(nn.Module):
    """Input transform: predicts a 3x3 matrix applied to the points."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv3 = nn.Conv1d(128, 1024, 1, bias=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) + torch.eye(3, device=x.device).view(1, 9)
        return x.view(-1, 3, 3)


class PointNetFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv3 = nn.Conv1d(128, 1024, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n = x.size(2)
        x = torch.bmm(x.transpose(2, 1), self.stn(x)).transpose(2, 1)
        point_feat = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(point_feat)))
        x = self.bn3(self.conv3(x)).max(dim=2, keepdim=True)[0]
        return torch.cat([x.repeat(1, 1, n), point_feat], 1)        # (B, 1088, N)


class PointNetSeg(nn.Module):
    def __init__(self, num_classes=8, **_):
        super().__init__()
        self.feat = PointNetFeat()
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    HEAD = ("conv3", "conv4", "bn3")
    LAST = "conv4"

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.conv4(x)

    def reset_head(self, num_classes):
        self.conv4 = nn.Conv1d(128, num_classes, 1).to(self.conv3.weight.device)


# --------------------------------------------------------------------------
# DGCNN (DGCNN1 in the original code)
# --------------------------------------------------------------------------

class DGCNNSeg(nn.Module):
    def __init__(self, num_classes=8, k=40, emb_dims=1024, dropout=0.5, **_):
        super().__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)
        act = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), self.bn1, act)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), self.bn2, act)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), self.bn3, act)
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), self.bn4, act)
        self.conv5 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), self.bn5, act)
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, 1, bias=False), self.bn6, act)
        self.conv7 = nn.Sequential(nn.Conv1d(emb_dims + 192, 512, 1, bias=False), self.bn7, act)
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, 1, bias=False), self.bn8, act)
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Conv1d(256, num_classes, 1, bias=False)

    HEAD = ("conv8", "conv9", "bn8")
    LAST = "conv9"

    def forward(self, x):
        n = x.size(2)
        x1 = self.conv2(self.conv1(get_graph_feature(x, self.k))).max(dim=-1)[0]
        x2 = self.conv4(self.conv3(get_graph_feature(x1, self.k))).max(dim=-1)[0]
        x3 = self.conv5(get_graph_feature(x2, self.k)).max(dim=-1)[0]
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv6(x).max(dim=-1, keepdim=True)[0]
        x = torch.cat((x.repeat(1, 1, n), x1, x2, x3), dim=1)
        x = self.dp1(self.conv8(self.conv7(x)))
        return self.conv9(x)

    def reset_head(self, num_classes):
        self.conv9 = nn.Conv1d(256, num_classes, 1, bias=False).to(self.conv8[0].weight.device)


# --------------------------------------------------------------------------
# CNN-Residual (CNN_residual_with_optimized_parameter in the original code)
# --------------------------------------------------------------------------

class CNNResidualSeg(nn.Module):
    """Six residual blocks (64-128-256-512-512-512) of per-point 1x1 convolutions."""

    def __init__(self, num_classes=8, emb_dims=1024, dropout=0.1889, negative_slope=0.042, **_):
        super().__init__()
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        widths = [3, 64, 128, 256, 512, 512, 512]
        for i in range(6):
            setattr(self, "block%d" % (i + 1), ResidualBlock(widths[i], widths[i + 1], self.activation))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False), nn.BatchNorm2d(256), self.activation)
        self.conv6 = nn.Sequential(nn.Conv1d(256, emb_dims, 1, bias=False), nn.BatchNorm1d(emb_dims),
                                   self.activation)
        self.conv7 = nn.Sequential(nn.Conv1d(emb_dims, 512, 1, bias=False), nn.BatchNorm1d(512),
                                   self.activation)
        self.dp1 = nn.Dropout(p=dropout)
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, 1, bias=False), nn.BatchNorm1d(256),
                                   self.activation)
        self.dp2 = nn.Dropout(p=dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 128, 1, bias=False), self.activation)
        self.conv10 = nn.Conv1d(128, num_classes, 1, bias=False)

    HEAD = ("conv9", "conv10")
    LAST = "conv10"

    def forward(self, x):
        x = x.unsqueeze(-1)
        for i in range(1, 7):
            x = getattr(self, "block%d" % i)(x)
        x = self.conv5(x).squeeze(-1)
        x = self.dp1(self.conv7(self.conv6(x)))
        x = self.dp2(self.conv8(x))
        return self.conv10(self.conv9(x))

    def reset_head(self, num_classes):
        self.conv10 = nn.Conv1d(128, num_classes, 1, bias=False).to(self.conv9[0].weight.device)
