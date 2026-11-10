#!/usr/bin/env python3
"""Convert labelled .pcd files to the Nx4 float32 [x, y, z, label] .npy format.

    python tools/pcd_to_npy.py raw/real/session1/ --out data/real/session1/
    # a recording whose AGV / Unlabelled ids are swapped relative to sim2real/labels.py
    python tools/pcd_to_npy.py raw/newlab/ --out data/real/newlab/ --label-map 6:7 7:6

Coordinates are written unchanged (metres, sensor frame); normalization
happens in the data loader. Reads ascii and binary PCD with a field named
`label`, `labels`, `class` or `semantic` (or --label-field).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim2real.io import expand_paths, label_lookup, read_pcd  # noqa: E402
from sim2real.labels import CLASS_NAMES, NUM_CLASSES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help=".pcd files, folders or glob patterns")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--label-map", nargs="*", default=[], metavar="SRC:DST",
                    help="remap label ids, e.g. 6:7 7:6")
    ap.add_argument("--label-field", help="name of the label field (default: auto)")
    a = ap.parse_args()

    files = expand_paths(a.paths, (".pcd",))
    if not files:
        sys.exit("no .pcd files found")
    lut = label_lookup(a.label_map)
    os.makedirs(a.out, exist_ok=True)

    totals = np.zeros(NUM_CLASSES, dtype=np.int64)
    for f in files:
        pts, lab = read_pcd(f, a.label_field)
        if lab is None:
            sys.exit("%s: no label field" % f)
        lab = lut[lab]
        bad = (lab < 0) | (lab >= NUM_CLASSES)
        if bad.any():
            sys.exit("%s: labels outside 0..%d: %s" % (f, NUM_CLASSES - 1, np.unique(lab[bad])))
        name = os.path.splitext(os.path.basename(f))[0] + ".npy"
        np.save(os.path.join(a.out, name), np.column_stack([pts, lab]).astype(np.float32))
        totals += np.bincount(lab, minlength=NUM_CLASSES)

    print("%d file(s) -> %s\n" % (len(files), a.out))
    for name, count in zip(CLASS_NAMES, totals):
        print("  %-13s %10d  %6.2f%%" % (name, count, 100.0 * count / max(totals.sum(), 1)))


if __name__ == "__main__":
    main()
