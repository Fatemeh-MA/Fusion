#!/usr/bin/env python3
"""Evaluate a checkpoint on every point of every frame in a split.

    python evaluate.py --ckpt outputs/train/best.pth --split splits/test.txt
    python evaluate.py --ckpt outputs/train/best.pth --split splits/test.txt \
        --save-predictions outputs/test_predictions

Checkpoints from the original training code (main2Stream.py) are accepted
too; they were fine-tuned and deployed on raw sensor coordinates, so no
normalization is applied.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from sim2real.data import read_split
from sim2real.models import output_layer  # noqa: E402
from sim2real.inference import evaluate_paths, load_checkpoint, untrained_class
from sim2real.labels import CLASS_NAMES, UNLABELLED


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, help="split file, one .npy path per line")
    ap.add_argument("--data-root", default="data", help="split entries are relative to this")
    ap.add_argument("--num-points", type=int, default=10500, help="points per forward pass")
    ap.add_argument("--ignore-index", type=int, default=UNLABELLED,
                    help="label excluded from scoring (-1 to score all classes)")
    ap.add_argument("--min-points", type=int, default=100,
                    help="classes with fewer ground-truth points in the split are not "
                         "averaged into mIoU/mAcc (default 100)")
    ap.add_argument("--save-predictions", metavar="DIR",
                    help="write Nx5 [x, y, z, label, pred] .npy files here")
    ap.add_argument("--out", help="write metrics as JSON to this file")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, ckpt = load_checkpoint(args.ckpt, device)
    if "normalization" not in ckpt:
        print("original-format checkpoint: evaluating on raw coordinates")
    print(repr(normalizer))

    paths = read_split(args.split, args.data_root)
    ignore = None if args.ignore_index < 0 else args.ignore_index
    num_classes = output_layer(model).out_channels

    on_frame = None
    if args.save_predictions:
        os.makedirs(args.save_predictions, exist_ok=True)

        def on_frame(path, points, labels, pred):
            name = os.path.splitext(os.path.basename(path))[0] + "_pred.npy"
            out = np.column_stack([points, labels, pred]).astype(np.float32)
            np.save(os.path.join(args.save_predictions, name), out)

    t0 = time.time()
    cm = evaluate_paths(model, paths, args.num_points, normalizer, device, num_classes,
                        ignore, on_frame, exclude_class=untrained_class(ckpt))
    cm.min_points = args.min_points
    seconds = (time.time() - t0) / max(len(paths), 1)

    s = cm.summary()
    ious = cm.iou()
    print("\n%d frames, %.3f s/frame" % (len(paths), seconds))
    print("mIoU %.4f  OA %.4f  mAcc %.4f\n" % (s["mIoU"], s["OA"], s["mAcc"]))
    print("%-13s %8s %10s" % ("class", "IoU", "points"))
    for c, name in enumerate(CLASS_NAMES[:num_classes]):
        if c == ignore:
            print("%-13s %8s %10s" % (name, "-", "ignored"))
            continue
        iou = "-" if np.isnan(ious[c]) else "%.4f" % ious[c]
        print("%-13s %8s %10d" % (name, iou, cm.matrix[c].sum()))

    if args.out:
        result = dict(s, frames=len(paths), seconds_per_frame=seconds,
                      iou={n: (None if np.isnan(v) else float(v))
                           for n, v in zip(CLASS_NAMES, ious)},
                      confusion_matrix=cm.matrix.tolist(), checkpoint=args.ckpt,
                      split=args.split)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print("\nmetrics written to %s" % args.out)


if __name__ == "__main__":
    main()
