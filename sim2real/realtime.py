"""Real-time segmentation of a live point cloud, independent of ROS.

    seg = Segmenter("outputs/finetune_labE/best.pth", crop_box=..., voxel_size=0.048)
    points, labels = seg(cloud_xyz)          # cloud_xyz: (N, 3) in the sensor frame

The pre-processing reproduces how the real frames of the dataset were made
(crop box in the sensor frame, voxel down-sampling), then applies the
normalization stored in the checkpoint. For the floor-aligned normalization
the floor is fitted once and re-used, because the sensors do not move; call
`reset_floor()` after moving them.
"""

import time

import numpy as np
import torch

from .data import estimate_floor, voxel_downsample
from .inference import load_checkpoint, predict_frame, untrained_class


def crop(points, box):
    """Keep points inside box = [[xmin, xmax], [ymin, ymax], [zmin, zmax]] (sensor frame)."""
    if box is None:
        return points
    lo = np.array([b[0] for b in box])
    hi = np.array([b[1] for b in box])
    return points[np.all((points >= lo) & (points <= hi), axis=1)]


class Segmenter:
    def __init__(self, checkpoint, crop_box=None, voxel_size=0.048, max_points=10500,
                 device=None, min_floor_points=500):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.normalizer, ckpt = load_checkpoint(checkpoint, self.device)
        self.model.eval()
        self.legacy = "normalization" not in ckpt
        self.crop_box = crop_box
        # the checkpoint's voxel size wins: it is what the network was trained with
        self.voxel_size = self.normalizer.voxel or voxel_size
        self.max_points = max_points
        self.exclude = untrained_class(ckpt)
        self.min_floor_points = min_floor_points
        self.floor = None
        self.timing = {}

    def reset_floor(self):
        self.floor = None

    def preprocess(self, cloud):
        points = np.asarray(cloud, dtype=np.float32)[:, :3]
        points = points[np.isfinite(points).all(1)]
        return voxel_downsample(crop(points, self.crop_box), self.voxel_size)

    def __call__(self, cloud):
        t0 = time.perf_counter()
        points = self.preprocess(cloud)
        t1 = time.perf_counter()
        if len(points) == 0:
            return points, np.zeros(0, dtype=np.int64)
        if self.normalizer.mode == "floor" and self.floor is None and len(points) >= self.min_floor_points:
            self.floor = estimate_floor(points)
        labels = predict_frame(self.model, points, self.max_points, self.normalizer, self.device,
                               exclude_class=self.exclude, floor=self.floor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.timing = {"preprocess": t1 - t0, "inference": t2 - t1, "points": len(points)}
        return points, labels
