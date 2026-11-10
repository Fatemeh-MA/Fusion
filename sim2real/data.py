"""Frames, splits, normalization, augmentation and the PyTorch dataset.

Data flow for one training sample:

    load frame (metres, sensor frame)
      -> Normalizer.downsample voxel grid: the same point density everywhere
      -> Normalizer.align      level the floor, floor at z = 0, centre x/y   (metres)
      -> random crop / chunk   of `num_points`
      -> Normalizer.recenter   median x/y of the piece to the origin         (metres)
      -> augment               rotation about z, flip, scale, shift, jitter  (metres)
      -> Normalizer.finish     divide by one fixed scale
      -> tensor

`align` uses nothing but the frame itself, so the same code runs in real time
and the result does not depend on where the sensor is mounted, how high, how
tilted, or which part of the room is cropped.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
# Frames and splits
# --------------------------------------------------------------------------

def load_frame(path):
    """Read an Nx4 [x, y, z, label] .npy frame -> (points float32 Nx3, labels int64 N)."""
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("%s: expected an Nx4 array, got shape %s" % (path, arr.shape))
    return arr[:, :3].astype(np.float32), arr[:, 3].astype(np.int64)


def read_split(split_file, data_root):
    """Paths listed in a split file, one per line, relative to `data_root`."""
    paths = []
    with open(split_file) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                paths.append(os.path.join(data_root, line))
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("%d file(s) from %s not found, e.g. %s"
                                % (len(missing), split_file, missing[0]))
    return paths


# --------------------------------------------------------------------------
# Density: voxel grid
# --------------------------------------------------------------------------

def voxel_downsample(points, voxel_size, labels=None):
    """One point per occupied voxel: the centroid, and the majority label.

    Every frame, simulated, recorded or live, goes through the same grid, so
    the point density is the same for all of them.
    """
    if not voxel_size or len(points) == 0:
        return points if labels is None else (points, labels)
    idx = np.floor(points / voxel_size).astype(np.int64)
    idx -= idx.min(0)
    dims = idx.max(0) + 1
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    out = np.empty((len(counts), 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.bincount(inverse, weights=points[:, d], minlength=len(counts))
    out = (out / counts[:, None]).astype(np.float32)
    if labels is None:
        return out
    n_cls = int(labels.max()) + 1
    votes = np.zeros((len(counts), n_cls), dtype=np.int64)
    np.add.at(votes, (inverse, labels), 1)
    return out, votes.argmax(1).astype(labels.dtype)


# --------------------------------------------------------------------------
# Floor estimation
# --------------------------------------------------------------------------

def rotation_to_z(normal):
    """Rotation matrix that maps the unit vector `normal` onto +z."""
    n = normal / np.linalg.norm(normal)
    v = np.cross(n, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(v), n[2]
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def estimate_floor(points, max_tilt_deg=25.0, threshold=0.04, iterations=200, seed=0):
    """Fit the floor plane without labels.

    RANSAC on the lowest 30 % of the points, keeping only near-horizontal
    planes, then a least-squares refit on the inliers. Returns (R, h): the
    levelled cloud is `points @ R.T` with the floor at z = h. Falls back to no
    rotation and the 1st height percentile if no floor plane is found.
    """
    p = np.asarray(points, dtype=np.float64)
    low = p[p[:, 2] <= np.percentile(p[:, 2], 30)]
    if len(low) < 50:
        return np.eye(3), float(np.percentile(p[:, 2], 1))
    rng = np.random.default_rng(seed)
    # all hypotheses at once, scored on at most 2000 of the low points
    tri = low[rng.integers(len(low), size=(iterations, 3))]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    n[ok] /= norm[ok, None]
    ok &= np.abs(n[:, 2]) >= np.cos(np.radians(max_tilt_deg))
    if not ok.any():
        return np.eye(3), float(np.percentile(p[:, 2], 1))
    n, a = n[ok], tri[ok, 0]
    score = low[rng.choice(len(low), min(len(low), 2000), replace=False)]
    dist = np.abs(score @ n.T - np.sum(a * n, axis=1))          # (points, hypotheses)
    best = np.argmax((dist < threshold).sum(0))
    inliers = np.abs((low - a[best]) @ n[best]) < threshold
    if inliers.sum() < 30:
        return np.eye(3), float(np.percentile(p[:, 2], 1))
    q = low[inliers]
    centroid = q.mean(0)
    d = q - centroid
    n = np.linalg.eigh(d.T @ d)[1][:, 0]     # normal = direction of least variance
    if n[2] < 0:
        n = -n
    R = rotation_to_z(n)
    return R, float((centroid @ R.T)[2])


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

class Normalizer:
    """Maps sensor-frame points (metres) to network input.

    mode "floor" (default): level the floor and put it at z = 0, subtract the
        frame's median x/y, divide by `scale`. Independent of the sensor pose
        and of the crop box; only `scale` is learnt from the training split.
    mode "global": (p - center) / scale with dataset-wide statistics.
    mode "none": raw coordinates, as used by the checkpoints of the paper.
    """

    MODES = ("floor", "global", "none")

    def __init__(self, mode="floor", center=(0.0, 0.0, 0.0), scale=1.0, voxel=None):
        if mode not in self.MODES:
            raise ValueError("normalization mode must be one of %s" % (self.MODES,))
        self.mode = mode
        self.center = np.asarray(center, dtype=np.float32).reshape(3)
        self.scale = float(scale)
        self.voxel = voxel            # metres; None = use the points as they are

    def downsample(self, points, labels=None):
        """Bring a frame to the density the network was trained with."""
        return voxel_downsample(points, self.voxel, labels)

    @classmethod
    def identity(cls):
        return cls("none")

    def align(self, points, floor=None):
        """Metric alignment. `floor=(R, h)` reuses a floor estimated earlier
        (e.g. once at start-up for a fixed sensor); otherwise it is fitted."""
        if self.mode == "none":
            return np.asarray(points, dtype=np.float32)
        if self.mode == "global":
            return (points - self.center).astype(np.float32)
        R, h = floor if floor is not None else estimate_floor(points)
        p = np.asarray(points, dtype=np.float64) @ R.T
        p[:, 2] -= h
        p[:, :2] -= np.median(p[:, :2], axis=0)
        return p.astype(np.float32)

    def recenter(self, points):
        """Median x/y of the piece the network will see -> origin.

        Applied to every training piece and to every inference chunk, so the
        input never depends on where in the room the piece was taken (the
        crop box of a new installation included). Floor mode only: "global"
        is defined by dataset-wide statistics and "none" is raw coordinates.
        """
        if self.mode != "floor":
            return np.asarray(points, dtype=np.float32)
        p = np.array(points, dtype=np.float32)
        p[:, :2] -= np.median(p[:, :2], axis=0)
        return p

    def finish(self, points):
        return (points / self.scale).astype(np.float32)

    def __call__(self, points, floor=None):
        return self.finish(self.recenter(self.align(points, floor)))

    @classmethod
    def fit(cls, paths, mode="floor", voxel=None):
        """Fit the statistics of `mode` on the training frames (streamed)."""
        if mode == "none":
            return cls("none", voxel=voxel)
        n, s, ss = 0, np.zeros(3), np.zeros(3)
        aligner = cls(mode)        # center 0 / identity scale, only used for "floor"
        for p in paths:
            pts = voxel_downsample(load_frame(p)[0], voxel).astype(np.float64)
            if mode == "floor":
                pts = aligner.align(pts).astype(np.float64)
            n += len(pts)
            s += pts.sum(0)
            ss += (pts ** 2).sum(0)
        if n == 0:
            raise ValueError("cannot fit normalization on an empty split")
        mean = s / n
        if mode == "floor":
            # RMS distance from the origin of the aligned frames
            return cls("floor", scale=float(np.sqrt(ss.sum() / n)), voxel=voxel)
        return cls("global", mean, float(np.sqrt((ss / n - mean ** 2).sum())), voxel=voxel)

    def to_dict(self):
        return {"mode": self.mode, "center": self.center.tolist(), "scale": self.scale,
                "voxel": self.voxel}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("mode", "global"), d["center"], d["scale"], d.get("voxel"))

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def __repr__(self):
        vox = ", voxel=%s" % self.voxel if self.voxel else ""
        if self.mode == "floor":
            return "Normalizer(mode=floor, scale=%.4f%s)" % (self.scale, vox)
        return "Normalizer(mode=%s, center=%s, scale=%.4f%s)" % (
            self.mode, np.round(self.center, 4).tolist(), self.scale, vox)


# --------------------------------------------------------------------------
# Augmentation (in metres)
# --------------------------------------------------------------------------

def nearest_chunk(points, seed_index, k):
    """Indices of the k points closest (in x/y) to points[seed_index].

    A contiguous piece of the scene at full density, unlike a random subset.
    """
    d = np.sum((points[:, :2] - points[seed_index, :2]) ** 2, axis=1)
    return np.argpartition(d, k - 1)[:k]


def random_crop(points, rng, size, min_points):
    """Indices of the points inside a random axis-aligned x/y box.

    Imitates the crop box of the real-time node, whose size and position
    change from one installation to the next.
    """
    side = rng.uniform(*size, size=2)
    centre = points[rng.integers(len(points)), :2]
    keep = np.all(np.abs(points[:, :2] - centre) <= side / 2, axis=1)
    idx = np.flatnonzero(keep)
    return idx if len(idx) >= min_points else np.arange(len(points))


def augment(points, rng, rotate_z=(0.0, 360.0), flip=False, scale=(0.9, 1.1), shift_xy=0.0,
            jitter_std=0.01, jitter_clip=0.05, **_):
    """Rotation about the vertical axis, mirror, uniform scale, x/y shift and jitter."""
    if flip and rng.random() < 0.5:
        points = points * np.array([-1.0, 1.0, 1.0], dtype=np.float32)   # mirror image of the scene
    if rotate_z:
        a = np.radians(rng.uniform(*rotate_z))
        c, s = np.cos(a), np.sin(a)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        points = points @ rot.T
    if scale:
        points = points * rng.uniform(*scale)
    if shift_xy:
        points = points + np.r_[rng.uniform(-shift_xy, shift_xy, 2), 0.0]
    if jitter_std:
        noise = np.clip(rng.normal(0.0, jitter_std, points.shape), -jitter_clip, jitter_clip)
        points = points + noise
    return points.astype(np.float32)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class PointCloudDataset(Dataset):
    """One random contiguous piece of at most `num_points` points per frame and epoch.

    With `pad=False` a frame with fewer points is returned whole (batch size 1)
    instead of being padded with repeated points, which would change the
    neighbourhoods DGCNN sees compared to inference.

    Evaluation runs on every point of a frame instead, see
    `sim2real.inference.predict_frame`.
    """

    def __init__(self, paths, num_points, normalizer, augmentation=None, seed=0, pad=True):
        self.paths = list(paths)
        self.num_points = num_points
        self.normalizer = normalizer
        self.augmentation = augmentation
        self.pad = pad
        self.rng = np.random.default_rng(seed)
        self._floor = {}           # per-frame floor fit, computed once
        self._frames = {}          # per-frame voxel-grid result, computed once

    def __len__(self):
        return len(self.paths)

    def _aligned(self, idx):
        if idx not in self._frames:
            self._frames[idx] = self.normalizer.downsample(*load_frame(self.paths[idx]))
        points, labels = self._frames[idx]
        floor = None
        if self.normalizer.mode == "floor":
            if idx not in self._floor:
                self._floor[idx] = estimate_floor(points)
            floor = self._floor[idx]
        return self.normalizer.align(points, floor), labels

    def __getitem__(self, idx):
        points, labels = self._aligned(idx)
        aug = self.augmentation or {}
        if aug.get("crop_xy") and self.rng.random() < aug.get("crop_prob", 0.5):
            keep = random_crop(points, self.rng, aug["crop_xy"], self.num_points // 2)
            points, labels = points[keep], labels[keep]
        n = len(points)
        if n > self.num_points:            # contiguous region, keeps the density
            choice = self.rng.permutation(nearest_chunk(points, self.rng.integers(n), self.num_points))
        elif n < self.num_points and self.pad:
            choice = self.rng.choice(n, self.num_points, replace=True)
        else:
            choice = self.rng.permutation(n)
        points, labels = points[choice], labels[choice]
        points = self.normalizer.recenter(points)
        if self.augmentation:
            points = augment(points, self.rng, **self.augmentation)
        points = self.normalizer.finish(points)
        return torch.from_numpy(points), torch.from_numpy(labels)


def worker_init_fn(worker_id):
    """Give every DataLoader worker its own random stream, different each epoch."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
