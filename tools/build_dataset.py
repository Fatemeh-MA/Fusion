#!/usr/bin/env python3
"""Assemble the released dataset from the authors' raw recording folders.

    python tools/build_dataset.py tools/dataset_sources.yaml --out data

For every source in the YAML file, the matching .npy frames are read, their
labels are remapped to the class map in sim2real/labels.py, and they are
written to <out>/<name>/<frame>.npy. A manifest.csv records the source file,
the label map and the class counts of every released frame, so the release
can be traced back and rebuilt.

Source entry:

    - name: real/lab_2023-11-10        # output folder below --out
      path: New_Real_Lab/Train         # relative to source_root
      pattern: '^10\\.11\\.2023_\\d+\\.npy$'   # optional regex on the file name
      label_map: {6: 7}                # raw id -> released id, applied at once
      by_date: true                    # optional: one sub-folder per recording date
                                       # taken from a dd.mm.yyyy file-name prefix

Simulated frames are rebuilt at full density from the raw Isaac Sim output
(see tools/isaac_to_npy.py):

    - name: sim/isaac_2023-05-02
      type: isaac
      raw: Isaac-SIM-synthetic-data/Tests/02052023
      path: Tests/02052023/Down/SIMCorrect annotation/NPY-Downto11000   # hand-corrected
      voxel: 0.02
"""

import argparse
import csv
import glob
import hashlib
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim2real.labels import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from isaac_to_npy import convert as convert_isaac  # noqa: E402


def read_frame(path):
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float64)   # some raw files were saved as object arrays
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("expected Nx4, got %s" % (arr.shape,))
    return arr[:, :3].astype(np.float32), arr[:, 3].astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", help="YAML file describing the sources")
    ap.add_argument("--out", default="data")
    ap.add_argument("--dry-run", action="store_true", help="only report, write nothing")
    a = ap.parse_args()

    with open(a.sources) as fh:
        cfg = yaml.safe_load(fh)
    root = cfg["source_root"]

    rows, seen, written, problems = [], {}, set(), []
    for src in cfg["sources"]:
        if src.get("type") == "isaac":
            out_dir = os.path.join(a.out, src["name"])
            if a.dry_run:
                print("%-28s isaac rebuild (skipped in --dry-run)" % src["name"])
                continue
            for name, raw_name, recovered, counts in convert_isaac(
                    os.path.join(root, src["raw"]), os.path.join(root, src["path"]), out_dir,
                    voxel=src.get("voxel", 0.02)):
                out_rel = "%s/%s" % (src["name"], name)
                written.add(out_rel)
                rows.append([out_rel, "%s/%s + %s/%s" % (src["raw"], raw_name, src["path"], name),
                             "isaac ids + hand labels", int(counts.sum())] + counts.tolist())
            print("%-28s rebuilt from raw Isaac frames" % src["name"])
            continue
        lut = np.arange(256)
        for k, v in (src.get("label_map") or {}).items():
            lut[int(k)] = int(v)
        files = sorted(glob.glob(os.path.join(root, src["path"], "*.npy")))
        if src.get("pattern"):
            files = [f for f in files if re.search(src["pattern"], os.path.basename(f))]
        if not files:
            problems.append("%s: no files in %s" % (src["name"], src["path"]))
            continue

        totals = np.zeros(NUM_CLASSES, dtype=np.int64)
        for f in files:
            name = src["name"]
            if src.get("by_date"):
                m = re.match(r"^(\d\d)\.(\d\d)\.(\d{4})_", os.path.basename(f))
                if not m:
                    problems.append("%s: no dd.mm.yyyy_ prefix, skipped" % f)
                    continue
                name += "_%s-%s-%s" % (m.group(3), m.group(2), m.group(1))
            pts, lab = read_frame(f)
            if lab.min() < 0 or lab.max() >= len(lut):
                problems.append("%s: label %d outside 0..255" % (f, lab.min() if lab.min() < 0 else lab.max()))
                continue
            lab = lut[lab]
            bad = lab >= NUM_CLASSES
            if bad.any():
                problems.append("%s: %d points with unknown labels %s dropped"
                                % (f, bad.sum(), np.unique(lab[bad]).tolist()))
                pts, lab = pts[~bad], lab[~bad]
            counts = np.bincount(lab, minlength=NUM_CLASSES)
            totals += counts

            fp = hashlib.sha1(lab.astype(np.int16).tobytes()).hexdigest()
            clean = re.sub(r" \((\d+(st|nd|rd|th) |another )?copy\)(?=\.npy$)", "", os.path.basename(f))
            out_rel = os.path.join(name, clean).replace(os.sep, "/")
            if out_rel in written:      # a different frame already has the cleaned name
                out_rel = os.path.join(name, os.path.basename(f)).replace(os.sep, "/")
            if fp in seen:
                problems.append("%s duplicates %s (same label sequence), skipped" % (out_rel, seen[fp]))
                continue
            seen[fp] = out_rel
            written.add(out_rel)
            if not a.dry_run:
                os.makedirs(os.path.join(a.out, name), exist_ok=True)
                np.save(os.path.join(a.out, out_rel), np.column_stack([pts, lab]).astype(np.float32))
            rows.append([out_rel, os.path.relpath(f, root), src.get("label_map") or {}, len(lab)]
                        + counts.tolist())

        share = ", ".join("%s %.1f%%" % (CLASS_NAMES[c], 100.0 * totals[c] / totals.sum())
                          for c in range(NUM_CLASSES) if totals[c])
        print("%-28s %4d frames  %s" % (src["name"], len(files), share))

    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "source", "label_map", "points"] + list(CLASS_NAMES))
            w.writerows(rows)
        print("\n%d frames written to %s, see %s/manifest.csv" % (len(rows), a.out, a.out))
    if problems:
        print("\n%d problem(s):" % len(problems))
        for p in problems[:30]:
            print("  " + p)


if __name__ == "__main__":
    main()
