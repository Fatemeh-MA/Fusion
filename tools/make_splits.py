#!/usr/bin/env python3
"""Split frames into train / val / test WITHOUT leaking frames between splits.

Frames are scanned under --root (any depth). The first folder below --root is
the domain (e.g. `sim`, `real`), and each domain is split separately with the
same ratios.

Consecutive LiDAR frames are almost identical, so splitting single frames at
random still leaks. Use --group-regex to keep a whole recording together: the
first capture group of the regex, applied to the path relative to --root,
is the group id.

    # every recording session (sub-folder) is one group
    python tools/make_splits.py --root data --group-regex '^([^/]+/[^/]+)/'

    # blocks of 10 consecutive frames (by the last number in the file name)
    python tools/make_splits.py --root data --block-size 10

After splitting, every frame is fingerprinted by its label sequence. Two files
with the same fingerprint are the same frame (an augmented copy keeps the
point order and labels), and the script refuses to write splits that put such
a pair on both sides of a split.
"""

import argparse
import collections
import glob
import hashlib
import os
import re
import sys

import numpy as np


def fingerprint(path):
    labels = np.load(path, mmap_mode="r")[:, 3]
    return hashlib.sha1(np.ascontiguousarray(labels, dtype=np.int16).tobytes()).hexdigest()


def split_groups(groups, val, test, rng):
    groups = sorted(groups)
    rng.shuffle(groups)
    n = len(groups)
    n_test = int(round(n * test))
    n_val = int(round(n * val))
    if n >= 3:   # keep at least one group per non-empty split
        n_test = max(n_test, 1 if test > 0 else 0)
        n_val = max(n_val, 1 if val > 0 else 0)
    return {"test": groups[:n_test],
            "val": groups[n_test:n_test + n_val],
            "train": groups[n_test + n_val:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--only", nargs="*", default=[],
                    help="restrict to these sub-folders of --root, e.g. real/newlab")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--group-regex", help="capture group 1 = group id (default: one group per frame)")
    ap.add_argument("--block-size", type=int,
                    help="group consecutive frames of a folder: index // block-size, where the "
                         "index is the last number in the file name")
    ap.add_argument("--per-folder", action="store_true",
                    help="split every recording folder separately, so each session is in "
                         "train, val and test (default: split the groups of a whole domain)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="splits")
    ap.add_argument("--prefix", default="", help="prefix for the output files, e.g. finetune_")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.root, "**", "*.npy"), recursive=True))
    rel = [os.path.relpath(f, a.root).replace(os.sep, "/") for f in files]
    if a.only:
        rel = [r for r in rel if any(r.startswith(o.rstrip("/") + "/") for o in a.only)]
    if not rel:
        sys.exit("no .npy files found under %s" % a.root)

    pattern = re.compile(a.group_regex) if a.group_regex else None
    by_domain = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rel:
        domain = r.split("/", 1)[0] if "/" in r else "."
        if a.per_folder:
            domain = os.path.dirname(r) or "."
        if pattern:
            m = pattern.search(r)
            if not m:
                sys.exit("--group-regex does not match %s" % r)
            group = m.group(1)
        elif a.block_size:
            numbers = re.findall(r"\d+", os.path.basename(r))
            if not numbers:
                sys.exit("--block-size: no frame number in %s" % r)
            group = "%s#%d" % (os.path.dirname(r), int(numbers[-1]) // a.block_size)
        else:
            group = r
        by_domain[domain][group].append(r)

    rng = np.random.default_rng(a.seed)
    splits = {"train": [], "val": [], "test": []}
    for domain in sorted(by_domain):
        groups = by_domain[domain]
        assignment = split_groups(list(groups), a.val, a.test, rng)
        counts = {}
        for name, gs in assignment.items():
            frames = [f for g in gs for f in groups[g]]
            splits[name] += frames
            counts[name] = "%d frames / %d groups" % (len(frames), len(gs))
        print("%-10s train %-24s val %-24s test %s"
              % (domain, counts["train"], counts["val"], counts["test"]))

    # leak check
    owner = {}
    leaks = []
    for name in ("train", "val", "test"):
        for r in splits[name]:
            fp = fingerprint(os.path.join(a.root, r))
            if fp in owner and owner[fp][0] != name:
                leaks.append((owner[fp], (name, r)))
            owner.setdefault(fp, (name, r))
    if leaks:
        print("\nERROR: %d frame(s) appear in more than one split, e.g.:" % len(leaks))
        for (s1, f1), (s2, f2) in leaks[:10]:
            print("  %s:%s == %s:%s" % (s1, f1, s2, f2))
        sys.exit("Group these files together with --group-regex (or remove the copies).")

    os.makedirs(a.out, exist_ok=True)
    for name, frames in splits.items():
        path = os.path.join(a.out, "%s%s.txt" % (a.prefix, name))
        with open(path, "w") as fh:
            fh.write("".join(f + "\n" for f in sorted(frames)))
        print("wrote %s (%d frames)" % (path, len(frames)))


if __name__ == "__main__":
    main()
