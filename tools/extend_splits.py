#!/usr/bin/env python3
"""Extend existing splits to frames added in a later dataset version.

Frames already listed in the base splits keep their split. A new frame of a
recording session that already has split frames joins the split of the most
similar of them (closest robots / people / AGV), so near-identical neighbours
never end up on different sides. Sessions without any split frames are split
in blocks of consecutive frames (natural file-name order).

    python tools/extend_splits.py --root data --base splits --out splits/v1.1 \\
        --group sim=sim/ labA=real/labA labB=real/labB labC=real/labC \\
                labD=real/labD labE=real/labE demo=real/demo \\
        --block sim=10 --all-real real_all

The files are written as <out>/<group>_{train,val,test}.txt; with --all-real
the train and val frames of all real groups are also written to
<out>/<name>_{train,val}.txt.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_splits import fingerprint, split_groups  # noqa: E402

SPLITS = ("train", "val", "test")


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def moving_points(path, n=2000, seed=0):
    a = np.load(path)
    sel = np.isin(a[:, 3].astype(int), (0, 1, 6))
    pts = a[sel, :3] if sel.sum() >= 50 else a[:, :3]
    if len(pts) > n:
        pts = pts[np.random.default_rng(seed).choice(len(pts), n, replace=False)]
    return pts


def chamfer(a, ta, b, tb):
    return tb.query(a)[0].mean() + ta.query(b)[0].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--base", default="splits", help="folder with the existing split files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", nargs="+", required=True, metavar="NAME=PREFIX",
                    help="split group and the path prefix of its frames, relative to --root")
    ap.add_argument("--block", nargs="*", default=[], metavar="NAME=N",
                    help="block size for new sessions of a group (default 5)")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-real", metavar="NAME",
                    help="also write train/val of all groups under real/ as NAME_{train,val}.txt")
    a = ap.parse_args()

    blocks = {k: int(v) for k, v in (b.split("=") for b in a.block)}
    rng = np.random.default_rng(a.seed)
    files = sorted(os.path.relpath(f, a.root).replace(os.sep, "/")
                   for f in glob.glob(os.path.join(a.root, "**", "*.npy"), recursive=True))
    os.makedirs(a.out, exist_ok=True)
    real_all = {"train": [], "val": []}
    owner, leaks = {}, []

    for spec in a.group:
        name, prefix = spec.split("=", 1)
        split_of = {}
        for s in SPLITS:
            path = os.path.join(a.base, "%s_%s.txt" % (name, s))
            if os.path.exists(path):
                for line in open(path):
                    if line.strip():
                        split_of[line.strip()] = s
        frames = [f for f in files if f.startswith(prefix)]
        missing = set(split_of) - set(frames)
        if missing:
            sys.exit("%s: %d frame(s) of the base splits are not in %s, e.g. %s"
                     % (name, len(missing), a.root, sorted(missing)[0]))

        sessions = {}
        for f in frames:
            sessions.setdefault(os.path.dirname(f), []).append(f)
        n_near, n_block = 0, 0
        for sess in sorted(sessions):
            known = [f for f in sessions[sess] if f in split_of]
            new = sorted((f for f in sessions[sess] if f not in split_of), key=natural_key)
            if not new:
                continue
            if known:   # join the split of the most similar known frame
                ref = [moving_points(os.path.join(a.root, f)) for f in known]
                trees = [cKDTree(p) for p in ref]
                for f in new:
                    p = moving_points(os.path.join(a.root, f))
                    t = cKDTree(p)
                    d = [chamfer(p, t, q, tq) for q, tq in zip(ref, trees)]
                    split_of[f] = split_of[known[int(np.argmin(d))]]
                    n_near += 1
            else:       # a new session: blocks of consecutive frames
                size = blocks.get(name, 5)
                groups = {}
                for i, f in enumerate(new):
                    groups.setdefault(i // size, []).append(f)
                for s, gs in split_groups(list(groups), a.val, a.test, rng).items():
                    for g in gs:
                        for f in groups[g]:
                            split_of[f] = s
                n_block += len(new)

        counts = []
        for s in SPLITS:
            lst = sorted(f for f, v in split_of.items() if v == s)
            counts.append(len(lst))
            with open(os.path.join(a.out, "%s_%s.txt" % (name, s)), "w") as fh:
                fh.write("".join(f + "\n" for f in lst))
            if prefix.startswith("real/") and s in real_all:
                real_all[s] += lst
            for f in lst:
                fp = fingerprint(os.path.join(a.root, f))
                if fp in owner and owner[fp][0] != s:
                    leaks.append((owner[fp], (s, f)))
                owner.setdefault(fp, (s, f))
        print("%-6s train %4d  val %4d  test %4d   (new: %d by nearest frame, %d in new sessions)"
              % ((name,) + tuple(counts) + (n_near, n_block)))

    if a.all_real:
        for s, lst in real_all.items():
            with open(os.path.join(a.out, "%s_%s.txt" % (a.all_real, s)), "w") as fh:
                fh.write("".join(f + "\n" for f in sorted(lst)))
        print("%-6s train %4d  val %4d" % (a.all_real, len(real_all["train"]), len(real_all["val"])))
    if leaks:
        for (s1, f1), (s2, f2) in leaks[:10]:
            print("  %s:%s == %s:%s" % (s1, f1, s2, f2))
        sys.exit("ERROR: %d frame(s) appear in more than one split" % len(leaks))


if __name__ == "__main__":
    main()
