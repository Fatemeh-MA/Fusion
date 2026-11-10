#!/usr/bin/env python3
"""Export .npy frames to ascii PCD (x y z label object) for (re-)annotation.

    # frames of the dataset
    python tools/npy_to_pcd.py data/real/labE_2026-08-27/ --out to_annotate/
    # predictions written by evaluate.py --save-predictions: use the prediction
    # as the starting label, so the annotator only fixes the mistakes
    python tools/npy_to_pcd.py outputs/pred/ --out to_annotate/ --use-prediction

The layout matches the semantic segmentation editor used to label the
dataset. Convert the corrected files back with tools/pcd_to_npy.py.
"""

import argparse
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim2real.io import expand_paths, read_labelled, write_pcd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".npy files, folders or glob patterns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-prediction", action="store_true",
                    help="write column 4 (prediction) instead of column 3 (label)")
    a = ap.parse_args()

    files = expand_paths(a.paths, (".npy",))
    os.makedirs(a.out, exist_ok=True)
    for f in files:
        pts, lab, pred = read_labelled(f)
        labels = pred if a.use_prediction else lab
        if labels is None:
            sys.exit("%s: no %s column" % (f, "prediction" if a.use_prediction else "label"))
        write_pcd(os.path.join(a.out, os.path.splitext(os.path.basename(f))[0] + ".pcd"),
                  pts, labels)
    print("%d file(s) -> %s" % (len(files), a.out))


if __name__ == "__main__":
    main()
