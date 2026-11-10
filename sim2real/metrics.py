"""Segmentation metrics computed from an accumulated confusion matrix.

IoU is accumulated over all points of all frames (not averaged per frame),
which is the standard definition and the one used in the paper.
"""

import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes, ignore_index=None, min_points=1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.min_points = min_points    # classes with fewer ground-truth points are not averaged
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows: truth

    def update(self, pred, target):
        pred = np.asarray(pred).reshape(-1)
        target = np.asarray(target).reshape(-1)
        keep = (target >= 0) & (target < self.num_classes)
        if self.ignore_index is not None:
            keep &= target != self.ignore_index
        idx = target[keep] * self.num_classes + pred[keep]
        self.matrix += np.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)

    def _evaluated_classes(self):
        """Classes with at least `min_points` ground-truth points, minus the ignored one."""
        present = self.matrix.sum(1) >= max(self.min_points, 1)
        if self.ignore_index is not None:
            present[self.ignore_index] = False
        return present

    def iou(self):
        """Per-class IoU; NaN for classes absent from the ground truth."""
        tp = np.diag(self.matrix).astype(np.float64)
        union = self.matrix.sum(0) + self.matrix.sum(1) - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = tp / union
        iou[~self._evaluated_classes()] = np.nan
        return iou

    def class_accuracy(self):
        tp = np.diag(self.matrix).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = tp / self.matrix.sum(1)
        acc[~self._evaluated_classes()] = np.nan
        return acc

    def summary(self):
        total = self.matrix.sum()
        return {
            "mIoU": float(np.nanmean(self.iou())),
            "OA": float(np.trace(self.matrix) / total) if total else float("nan"),
            "mAcc": float(np.nanmean(self.class_accuracy())),
        }
