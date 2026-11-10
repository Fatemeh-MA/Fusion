"""Reading and writing labelled point clouds (.npy and .pcd)."""

import glob
import os

import numpy as np

PCD_TYPES = {("F", 4): "<f4", ("F", 8): "<f8",
             ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4",
             ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4"}
LABEL_FIELDS = ("label", "labels", "class", "semantic")


def read_pcd(path, label_field=None):
    """Read an ascii or binary PCD -> (points Nx3 float32, labels N int32 or None)."""
    with open(path, "rb") as fh:
        header, n_bytes, n_lines = {}, 0, 0
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("%s: no DATA line in header" % path)
            n_bytes += len(line)
            n_lines += 1
            txt = line.decode("ascii", "replace").strip()
            if not txt or txt.startswith("#"):
                continue
            key, _, val = txt.partition(" ")
            header[key.upper()] = val.strip()
            if key.upper() == "DATA":
                break

        fields = header["FIELDS"].split()
        sizes = [int(x) for x in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(x) for x in header.get("COUNT", " ".join("1" * len(fields))).split()]
        n = int(header.get("POINTS", header["WIDTH"]))
        kind = header["DATA"].lower()

        wanted = (label_field,) if label_field else LABEL_FIELDS
        lab_i = next((i for i, f in enumerate(fields) if f.lower() in wanted), None)
        xyz_i = [fields.index(c) for c in ("x", "y", "z")]

        if kind == "ascii":
            raw = np.loadtxt(path, skiprows=n_lines, ndmin=2)
            pts, lab = raw[:, xyz_i], (raw[:, lab_i] if lab_i is not None else None)
        elif kind == "binary":
            if any(c != 1 for c in counts):
                raise ValueError("%s: fields with COUNT > 1 are not supported" % path)
            dt = np.dtype([(f, PCD_TYPES[(t, s)]) for f, t, s in zip(fields, types, sizes)])
            fh.seek(n_bytes)
            arr = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
            pts = np.column_stack([arr[fields[i]] for i in xyz_i])
            lab = arr[fields[lab_i]] if lab_i is not None else None
        else:
            raise ValueError("%s: DATA %s not supported, re-save as ascii or binary" % (path, kind))

    return pts.astype(np.float32), (lab.astype(np.int32) if lab is not None else None)


def write_pcd(path, points, labels, objects=None):
    """Write an ascii PCD with fields x y z label object.

    This is the layout of the semantic segmentation editor used for labelling;
    `object` (instance id) is -1 when unknown.
    """
    n = len(points)
    objects = np.full(n, -1, dtype=np.int64) if objects is None else objects
    header = ("VERSION .7\nFIELDS x y z label object\nSIZE 4 4 4 4 4\nTYPE F F F I I\n"
              "COUNT 1 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA ascii\n"
              % (n, n))
    with open(path, "w") as fh:
        fh.write(header)
        np.savetxt(fh, np.column_stack([points, labels, objects]),
                   fmt=["%.6f", "%.6f", "%.6f", "%d", "%d"])


def read_labelled(path, label_field=None):
    """Read .npy ([x, y, z, label(, pred)]) or .pcd.

    Returns (points Nx3, labels N or None, predictions N or None).
    """
    if path.endswith(".pcd"):
        pts, lab = read_pcd(path, label_field)
        return pts, lab, None
    arr = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    lab = arr[:, 3].astype(np.int64) if arr.shape[1] > 3 else None
    pred = arr[:, 4].astype(np.int64) if arr.shape[1] > 4 else None
    return arr[:, :3], lab, pred


def expand_paths(items, extensions=(".npy", ".pcd")):
    """Files, folders and glob patterns -> sorted list of matching files."""
    out = []
    for item in items:
        if os.path.isdir(item):
            out += [os.path.join(item, f) for f in os.listdir(item) if f.endswith(extensions)]
        else:
            out += [f for f in glob.glob(item) if f.endswith(extensions)]
    return sorted(set(out))


def label_lookup(pairs, size=256):
    """["6:7", "7:6"] -> lookup array that applies all remappings at once."""
    lut = np.arange(size)
    for item in pairs or []:
        src, dst = (int(v) for v in item.split(":"))
        lut[src] = dst
    return lut
