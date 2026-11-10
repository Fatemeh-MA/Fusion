#!/usr/bin/env python3
"""Rebuild full-density labelled frames from raw NVIDIA Isaac Sim output.

    python tools/isaac_to_npy.py --raw "Isaac-SIM-synthetic-data/Tests/02052023" \
        --corrected "Tests/02052023/Down/SIMCorrect annotation/NPY-Downto11000" \
        --out data/sim/isaac_2023-05-02 --voxel 0.02

Raw frames (`roblabTotal <time>.npy`, x y z isaac_id) come with a
`Config <time>.txt` mapping Isaac's semantic ids to class names; the ids
change from frame to frame. Robots carry no semantic class in Isaac (id 0), so
they were labelled by hand on down-sampled copies of the frames (the
"corrected" folder; static classes kept 1:10, dynamic ones 1:2).

To obtain frames at the full density of the simulated LiDAR, this tool puts
the hand-made labels back onto the full raw frame and writes it on a uniform
voxel grid:

  1. find the raw frame the corrected copy was cut from (exact point match)
     and crop it to the region the corrected copy covers;
  2. raw points that are in the corrected copy keep its label;
  3. other points with a named Isaac class get that class;
  4. other id-0 points get the label of the nearest corrected id-0 point
     within --radius, else Unlabelled.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim2real.data import voxel_downsample  # noqa: E402
from sim2real.labels import CLASS_NAMES, NUM_CLASSES, UNLABELLED  # noqa: E402

ISAAC_NAMES = {"human": 1, "wall": 2, "floors": 3, "floor": 3, "demonstrator": 4,
               "table": 5, "agv": 6, "ceil": UNLABELLED}


def read_config(path):
    """{isaac_id: class index}; ids without a known name are left out."""
    out = {}
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\s*ID\s+(\d+)\s+Label\s*=\s*(.*)", line)
            if m and m.group(2).strip().lower() in ISAAC_NAMES:
                out[int(m.group(1))] = ISAAC_NAMES[m.group(2).strip().lower()]
    return out


def find_sources(raw_files, corrected, sample=200, seed=0):
    """For each corrected frame, the raw frame that contains all of its points."""
    rng = np.random.default_rng(seed)
    probes = {k: v[rng.choice(len(v), min(sample, len(v)), replace=False), :3]
              for k, v in corrected.items()}
    best = {k: (-1.0, None) for k in corrected}
    for r in raw_files:
        tree = cKDTree(np.asarray(np.load(r, allow_pickle=True), dtype=np.float64)[:, :3])
        for k, probe in probes.items():
            score = float((tree.query(probe)[0] < 1e-5).mean())
            if score > best[k][0]:
                best[k] = (score, r)
    return best


def relabel(raw, config, corr, radius):
    pts, ids = raw[:, :3], raw[:, 3].astype(np.int64)
    labels = np.full(len(pts), UNLABELLED, dtype=np.int64)
    named = np.isin(ids, list(config))
    labels[named] = [config[i] for i in ids[named]]

    d, idx = cKDTree(corr[:, :3]).query(pts)
    exact = d < 1e-5
    labels[exact] = corr[idx[exact], 3]                      # hand-made labels win

    # corrected points that sit on raw id 0: the hand-labelled robots, humans, ...
    corr_ids = ids[cKDTree(pts).query(corr[:, :3])[1]]
    anchors = corr[corr_ids == 0]
    todo = (ids == 0) & ~exact
    if len(anchors) and todo.any():
        d0, i0 = cKDTree(anchors[:, :3]).query(pts[todo])
        fill = np.where(d0 <= radius, anchors[i0, 3], UNLABELLED)
        labels[todo] = fill
    return pts.astype(np.float32), labels, exact.sum()


def convert(raw_dir, corrected_dir, out_dir, voxel=0.02, radius=0.1, min_match=0.99):
    corrected = {os.path.basename(f): np.load(f).astype(np.float64)
                 for f in sorted(glob.glob(os.path.join(corrected_dir, "*.npy")))}
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "roblabTotal*.npy")))
    sources = find_sources(raw_files, corrected)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, corr in corrected.items():
        score, raw_path = sources[name]
        if score < min_match:
            print("  %s: no raw frame found (best match %.2f), skipped" % (name, score))
            continue
        stamp = os.path.basename(raw_path)[len("roblabTotal "):-len(".npy")]
        cfg_path = os.path.join(raw_dir, "Config %s.txt" % stamp)
        config = read_config(cfg_path) if os.path.exists(cfg_path) else {}
        raw = np.asarray(np.load(raw_path, allow_pickle=True), dtype=np.float64)
        lo, hi = corr[:, :3].min(0) - 0.05, corr[:, :3].max(0) + 0.05
        raw = raw[np.all((raw[:, :3] >= lo) & (raw[:, :3] <= hi), axis=1)]
        pts, lab, n_exact = relabel(raw, config, corr, radius)
        pts, lab = voxel_downsample(pts, voxel, lab)
        np.save(os.path.join(out_dir, name), np.column_stack([pts, lab]).astype(np.float32))
        rows.append((name, os.path.basename(raw_path), n_exact / len(corr), np.bincount(lab, minlength=NUM_CLASSES)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder with roblabTotal*.npy and Config*.txt")
    ap.add_argument("--corrected", required=True, help="hand-corrected down-sampled frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.02, help="output voxel grid (metres)")
    ap.add_argument("--radius", type=float, default=0.1,
                    help="max distance for copying a hand-made label to an id-0 point")
    a = ap.parse_args()
    rows = convert(a.raw, a.corrected, a.out, a.voxel, a.radius)
    totals = sum(r[3] for r in rows)
    print("%d frames -> %s (corrected points recovered: min %.3f)" % (
        len(rows), a.out, min(r[2] for r in rows) if rows else 0))
    for name, count in zip(CLASS_NAMES, totals):
        print("  %-13s %10d  %6.2f%%" % (name, count, 100.0 * count / max(totals.sum(), 1)))


if __name__ == "__main__":
    main()
