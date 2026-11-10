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
      rename: ['^[A-Z]+', '']          # optional: regex and replacement for file names
      source_label: recording 1        # optional: shown in manifest.csv instead of the path
      dedupe: true                     # optional: skip frames already in the dataset at
                                       # another resolution or with older labels

COVERED scenes (6 classes) get the classes added later from relabelled frames
of the same room:

    - name: real/labA_covered_s1
      type: covered
      path: COVERED/Scenario1/npy_label_data
      reference: real/labA_NewLabelRealDown   # released folder with the added classes

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
from sim2real.data import voxel_downsample  # noqa: E402
from isaac_to_npy import convert as convert_isaac  # noqa: E402


def read_frame(path):
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float64)   # some raw files were saved as object arrays
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("expected Nx4, got %s" % (arr.shape,))
    return arr[:, :3].astype(np.float32), arr[:, 3].astype(np.int64)


COVERED_MAP = {1: 3, 2: 2, 3: 0, 4: 1, 5: 6}      # COVERED v1 -> this dataset; 0 = "Unlabeled"
STATIC = (2, 3, 4, 5, 7)                           # wall, floor, demonstrator, table, unlabelled


def covered_labels(pts, lab, tree, ref_lab, radius=0.05):
    """COVERED labels -> this dataset. Points COVERED leaves unlabelled (0) take
    the static class (demonstrator, table, ...) of the nearest point of the
    relabelled frames of the same room within `radius`, else Unlabelled."""
    out = np.full(len(lab), 7, dtype=np.int64)
    for src, dst in COVERED_MAP.items():
        out[lab == src] = dst
    zero = np.flatnonzero(lab == 0)
    if len(zero):
        d, j = tree.query(pts[zero], distance_upper_bound=radius)
        ok = np.isfinite(d)
        out[zero[ok]] = ref_lab[j[ok]]
    return out


def dynamic_keys(pts, lab):
    """Hashed coordinates (0.1 mm) of the moving objects of a frame.

    Robots, people and AGV together (all points if there are fewer than 50).
    Static scene parts are identical in every simulated frame, so they cannot
    tell frames apart, and a person may stand still while the robot moves.
    """
    sel = np.isin(lab, (0, 1, 6))
    if sel.sum() < 50:
        sel = np.ones(len(lab), bool)
    q = np.round(pts[sel].astype(np.float64) * 1e4).astype(np.int64)
    return np.unique((q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791))


class FrameIndex:
    """Finds a frame already in the dataset that shares most moving points."""

    def __init__(self):
        self.owner, self.size, self.names = {}, [], []

    def add(self, pts, lab, name):
        keys = dynamic_keys(pts, lab)
        fid = len(self.size)
        self.size.append(len(keys))
        self.names.append(name)
        for k in keys.tolist():
            self.owner.setdefault(k, []).append(fid)

    def duplicate_of(self, pts, lab, share=0.9):
        # another version of the same frame (other resolution or labelling)
        # shares ~100 % of the moving points; distinct frames of the static
        # 2022 simulation share at most ~60 %, real frames practically none
        keys = dynamic_keys(pts, lab)
        hits = {}
        for k in keys.tolist():
            for fid in self.owner.get(k, ()):
                hits[fid] = hits.get(fid, 0) + 1
        for fid, n in sorted(hits.items(), key=lambda kv: -kv[1])[:1]:
            if n >= share * min(len(keys), self.size[fid]):
                return self.names[fid]
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", help="YAML file describing the sources")
    ap.add_argument("--out", default="data")
    ap.add_argument("--dry-run", action="store_true", help="only report, write nothing")
    ap.add_argument("--problems", help="write the full list of skipped files to this file")
    a = ap.parse_args()

    with open(a.sources) as fh:
        cfg = yaml.safe_load(fh)
    root = cfg["source_root"]

    rows, seen, written, problems = [], {}, set(), []
    index = FrameIndex()
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
                arr = np.load(os.path.join(out_dir, name))
                index.add(arr[:, :3], arr[:, 3].astype(np.int64), out_rel)
                rows.append([out_rel, "%s/%s + %s/%s" % (src["raw"], raw_name, src["path"], name),
                             "isaac ids + hand labels", int(counts.sum())] + counts.tolist())
            print("%-28s rebuilt from raw Isaac frames" % src["name"])
            continue
        tree = None
        if src.get("type") == "covered":
            from scipy.spatial import cKDTree
            ref = [np.load(f) for f in sorted(glob.glob(os.path.join(a.out, src["reference"], "*.npy")))]
            ref = np.vstack(ref)
            ref = ref[np.isin(ref[:, 3], STATIC)]
            # one point per 1 cm voxel (majority label): the frames of a room
            # repeat the same static points, and the query is much faster
            ref_pts, ref_lab = voxel_downsample(ref[:, :3], 0.01, ref[:, 3].astype(np.int64))
            tree = cKDTree(ref_pts)
        lut = np.arange(256)
        for k, v in (src.get("label_map") or {}).items():
            lut[int(k)] = int(v)
        files = sorted(glob.glob(os.path.join(root, src["path"], "*.npy")))
        n_added = 0
        if src.get("pattern"):
            files = [f for f in files if re.search(src["pattern"], os.path.basename(f))]
        if not files:
            problems.append("%s: no files in %s" % (src["name"], src["path"]))
            continue

        totals = np.zeros(NUM_CLASSES, dtype=np.int64)
        for f in files:
            name = src["name"]
            base = os.path.basename(f)
            if src.get("rename"):
                base = re.sub(src["rename"][0], src["rename"][1], base)
            if src.get("by_date"):
                m = re.match(r"^(\d\d)\.(\d\d)\.(\d{4})_", base)
                if not m:
                    problems.append("%s: no dd.mm.yyyy_ prefix, skipped" % f)
                    continue
                name += "_%s-%s-%s" % (m.group(3), m.group(2), m.group(1))
            try:
                pts, lab = read_frame(f)
            except ValueError as err:
                problems.append("%s: %s, skipped" % (f, err))
                continue
            if tree is not None:
                lab = covered_labels(pts, lab, tree, ref_lab)
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

            fp = hashlib.sha1(lab.astype(np.int16).tobytes()).hexdigest()
            clean = re.sub(r" \((\d+(st|nd|rd|th) |another )?copy\)(?=\.npy$)", "", base)
            out_rel = os.path.join(name, clean).replace(os.sep, "/")
            if out_rel in written:      # a different frame already has the cleaned name
                out_rel = os.path.join(name, base).replace(os.sep, "/")
            stem, suffix = out_rel[:-4], 1
            while out_rel in written:   # still taken: number it
                suffix += 1
                out_rel = "%s_%d.npy" % (stem, suffix)
            if fp in seen:
                problems.append("%s duplicates %s (same label sequence), skipped" % (out_rel, seen[fp]))
                continue
            if src.get("dedupe"):
                twin = index.duplicate_of(pts, lab)
                if twin:
                    problems.append("%s is %s at another resolution/labelling, skipped" % (out_rel, twin))
                    continue
            seen[fp] = out_rel
            written.add(out_rel)
            index.add(pts, lab, out_rel)
            n_added += 1
            totals += counts
            if not a.dry_run:
                os.makedirs(os.path.join(a.out, name), exist_ok=True)
                np.save(os.path.join(a.out, out_rel), np.column_stack([pts, lab]).astype(np.float32))
            source = ("%s/%s" % (src["source_label"], base)) if src.get("source_label") \
                else os.path.relpath(f, root)
            how = "COVERED + static classes of %s" % src["reference"] if tree is not None \
                else (src.get("label_map") or {})
            rows.append([out_rel, source, how, len(lab)] + counts.tolist())

        share = ", ".join("%s %.1f%%" % (CLASS_NAMES[c], 100.0 * totals[c] / max(totals.sum(), 1))
                          for c in range(NUM_CLASSES) if totals[c])
        print("%-28s %4d frames  %s" % (src["name"], n_added, share))

    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "source", "label_map", "points"] + list(CLASS_NAMES))
            w.writerows(rows)
        print("\n%d frames written to %s, see %s/manifest.csv" % (len(rows), a.out, a.out))
    if problems:
        kinds = {"same label sequence": "exact copies", "another resolution": "other versions of a frame"}
        for key, label in kinds.items():
            print("skipped, %s: %d" % (label, sum(key in p for p in problems)))
        print("\n%d problem(s):" % len(problems))
        if a.problems:
            with open(a.problems, "w") as fh:
                fh.write("".join(p + "\n" for p in problems))
        for p in problems[:15]:
            print("  " + p)


if __name__ == "__main__":
    main()
