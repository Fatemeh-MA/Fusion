#!/usr/bin/env python3
"""Browse labelled frames or predictions (.npy or .pcd) with Open3D.

    python tools/visualize.py data/real/labE_2026-08-27/
    python tools/visualize.py "raw/*.pcd" --label-map 6:7 7:6 --voxel 0.02
    python tools/visualize.py outputs/pred/ --pred          # files from evaluate.py --save-predictions
    python tools/visualize.py data/sim/isaac_2023-05-02/ --browser   # Open3D-ML browser with a file list

Accepts Nx4 [x, y, z, label] and Nx5 [x, y, z, label, pred] .npy files and
labelled PCD files; folders and glob patterns are expanded.

Keys in the default viewer:  N / P  next / previous    L  label <-> prediction    Q  quit
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim2real.io import expand_paths, label_lookup, read_labelled  # noqa: E402
from sim2real.labels import CLASS_COLORS, CLASS_NAMES  # noqa: E402

try:
    import open3d as o3d
except ImportError:
    sys.exit("this tool needs Open3D: pip install open3d")

COLORS = np.array(CLASS_COLORS)


def load(path, lut, voxel, label_field):
    pts, lab, pred = read_labelled(path, label_field)
    lab = lut[lab] if lab is not None else np.full(len(pts), 7)
    if voxel:
        # keep the label of one point per voxel
        key = np.floor(pts / voxel).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        pts, lab = pts[first], lab[first]
        pred = pred[first] if pred is not None else None
    return pts, lab, pred


def browser(files, lut, voxel, label_field):
    """Open3D-ML visualizer: all frames in one window, selectable from a list."""
    import open3d.ml.torch as ml3d
    lut_vis = ml3d.vis.LabelLUT()
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        lut_vis.add_label(name, i, list(color))
    items = []
    for f in files:
        pts, lab, pred = load(f, lut, voxel, label_field)
        item = {"name": os.path.basename(f), "points": pts, "labels": lab}
        if pred is not None:
            item["pred"] = pred
        items.append(item)
    vis = ml3d.vis.Visualizer()
    vis.set_lut("labels", lut_vis)
    vis.set_lut("pred", lut_vis)
    vis.visualize(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="files, folders or glob patterns")
    ap.add_argument("--pred", action="store_true", help="start by colouring the prediction")
    ap.add_argument("--label-map", nargs="*", default=[], metavar="SRC:DST",
                    help="remap labels before display, e.g. 6:7 7:6")
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel down-sampling (metres)")
    ap.add_argument("--label-field", help="PCD label field name (default: auto)")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--browser", action="store_true", help="use the Open3D-ML visualizer")
    a = ap.parse_args()

    files = expand_paths(a.paths)
    if not files:
        sys.exit("no .npy/.pcd files found")
    lut = label_lookup(a.label_map)
    print("legend: " + ", ".join("%d %s" % (i, n) for i, n in enumerate(CLASS_NAMES)))
    if a.browser:
        return browser(files, lut, a.voxel, a.label_field)

    state = {"i": 0, "pred": a.pred}
    pcd = o3d.geometry.PointCloud()

    def show():
        pts, lab, pred = load(files[state["i"]], lut, a.voxel, a.label_field)
        use_pred = state["pred"] and pred is not None
        labels = np.clip(pred if use_pred else lab, 0, len(COLORS) - 1)
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(COLORS[labels])
        present = ", ".join(CLASS_NAMES[c] for c in np.unique(labels))
        print("[%d/%d] %s  (%s)  %s" % (state["i"] + 1, len(files), files[state["i"]],
                                        "prediction" if use_pred else "label", present))

    def step(d):
        def cb(vis):
            state["i"] = (state["i"] + d) % len(files)
            show()
            vis.update_geometry(pcd)
            return False
        return cb

    def toggle(vis):
        state["pred"] = not state["pred"]
        show()
        vis.update_geometry(pcd)
        return False

    show()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("sim2real", 1280, 800)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = a.point_size
    vis.register_key_callback(ord("N"), step(1))
    vis.register_key_callback(ord("P"), step(-1))
    vis.register_key_callback(ord("L"), toggle)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
