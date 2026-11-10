#!/usr/bin/env python3
"""Cut the work cell out of labelled frames with an axis-aligned box.

    python tools/crop_frames.py raw_npy/ --out cropped/ \
        --box -0.2 2.5 -3.2 0.0 -2.0 0.5

    # box defined in a frame rotated about the sensor's y axis (sensor mounted
    # at an angle); the output stays in the sensor frame
    python tools/crop_frames.py raw_npy/ --out cropped/ --box -2.3 2.66 -2.8 0 -10 0.4 --rotate-y -44

Use the same box as `crop_box` in configs/realtime.yaml, so that training
frames and live frames cover the same region.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim2real.io import expand_paths, read_labelled  # noqa: E402
from sim2real.data import voxel_downsample  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".npy/.pcd files, folders or glob patterns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", nargs=6, type=float, required=True,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    ap.add_argument("--rotate-y", type=float, default=0.0,
                    help="degrees; the box is applied after rotating the points about y")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="optional voxel down-sampling of unlabelled clouds (metres)")
    a = ap.parse_args()

    t = np.radians(a.rotate_y)
    rot = np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])
    lo, hi = np.array(a.box[0::2]), np.array(a.box[1::2])
    os.makedirs(a.out, exist_ok=True)
    files = expand_paths(a.paths)
    for f in files:
        pts, lab, _ = read_labelled(f)
        q = pts @ rot.T if a.rotate_y else pts       # rot.T == the original inverse_matrix
        keep = np.all((q >= lo) & (q <= hi), axis=1)
        pts = pts[keep]
        if lab is None:
            if a.voxel:
                pts = voxel_downsample(pts, a.voxel)
            out = pts
        else:
            out = np.column_stack([pts, lab[keep]])
        name = os.path.splitext(os.path.basename(f))[0] + ".npy"
        np.save(os.path.join(a.out, name), out.astype(np.float32))
        print("%s: %d points kept" % (name, len(pts)))
    print("%d file(s) -> %s" % (len(files), a.out))


if __name__ == "__main__":
    main()
