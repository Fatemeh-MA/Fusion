#!/usr/bin/env python3
"""How much does accuracy drop when the sensor is installed somewhere else?

Every test frame is re-expressed in the frame of a sensor with a different
pose (moved, raised, tilted, turned) and evaluated again. A model that only
works for the pose it was trained with loses accuracy here; one that works in
a new installation does not.

    python tools/eval_sensor_moves.py --ckpt outputs/finetune_labD/best.pth \
        --split splits/labD_test.txt
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim2real.data import load_frame, read_split  # noqa: E402
from sim2real.models import output_layer  # noqa: E402
from sim2real.inference import load_checkpoint, predict_frame, untrained_class  # noqa: E402
from sim2real.labels import UNLABELLED  # noqa: E402
from sim2real.metrics import ConfusionMatrix  # noqa: E402

# name: (yaw deg, tilt deg, translation x y z in metres)
MOVES = {
    "as recorded":           (0, 0, (0.0, 0.0, 0.0)),
    "moved 2 m sideways":    (0, 0, (2.0, -1.0, 0.0)),
    "mounted 0.4 m higher":  (0, 0, (0.0, 0.0, -0.4)),
    "tilted 4 deg":          (0, 4, (0.0, 0.0, 0.0)),
    "turned 90 deg":         (90, 0, (0.0, 0.0, 0.0)),
    "all of the above":      (90, 4, (2.0, -1.0, -0.4)),
}


def move(points, yaw, tilt, t):
    y, a = np.radians(yaw), np.radians(tilt)
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return (points @ (rx @ rz).T + np.asarray(t)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--num-points", type=int, default=10500, help="points per forward pass")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, ckpt = load_checkpoint(a.ckpt, device)
    model.eval()
    exclude = untrained_class(ckpt)
    frames = [load_frame(p) for p in read_split(a.split, a.data_root)]
    print("%s  (%r, %d frames)\n" % (a.ckpt, normalizer, len(frames)))
    print("%-22s %8s %8s" % ("sensor pose", "mIoU", "OA"))
    for name, (yaw, tilt, t) in MOVES.items():
        cm = ConfusionMatrix(output_layer(model).out_channels, UNLABELLED)
        for i, (pts, lab) in enumerate(frames):
            moved, lab_v = normalizer.downsample(move(pts, yaw, tilt, t), lab)
            pred = predict_frame(model, moved, a.num_points, normalizer, device,
                                 seed=i, exclude_class=exclude)
            cm.update(pred, lab_v)
        s = cm.summary()
        print("%-22s %8.3f %8.3f" % (name, s["mIoU"], s["OA"]))


if __name__ == "__main__":
    main()
