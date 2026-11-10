# Dataset card

> Items marked **TODO** need to be filled in by the authors before release.

## Summary

LiDAR scans of human-robot collaboration workspaces with per-point semantic
labels: 1252 frames, 20 recording sessions, two domains (version 1.1).

| folder | domain | frames | used for |
|--------|--------|--------|----------|
| `sim/isaac_2022-10` | NVIDIA Isaac Sim 2022, October 2022 runs | 120 | pre-training |
| `sim/isaac_2023-05-02` | NVIDIA Isaac Sim 2022 | 126 | pre-training |
| `real/labA_NewLabelRealDown` | lab A, 2023 | 99 | fine-tuning / test |
| `real/labA_covered_s1` … `_s4` | lab A, COVERED scenarios 1–4 | 25 + 18 + 32 + 44 | fine-tuning / test |
| `real/labB_2023-10-22`, `_2023-10-31`, `_2023-11-01`, `_2023-11-10` | lab B | 61 + 183 + 109 + 106 | fine-tuning / test |
| `real/labC_2023-11-20` | lab C, with demonstrator | 50 | fine-tuning / test |
| `real/labD_2023-11-29` … `_2024-06-07` (5 sessions) | lab D | 73 | fine-tuning / test |
| `real/labE_2026-08-27` | lab E | 68 | fine-tuning / test (recorded after the paper) |
| `real/demo_2023-11-25`, `_2023-11-26` | demonstration cell, two days | 105 + 33 | fine-tuning / test |

### Versions

| version | frames | changes |
|---------|--------|---------|
| 1.0 | 761 | first release |
| 1.1 | 1252 | adds the October 2022 simulation runs, the COVERED scenarios of lab A with AGV, table and demonstrator labels, the demonstration recording, and further labelled frames of labs A, B, D and E. The 761 files of version 1.0 are unchanged. |

TODO: lab names (the paper's "Lab 1" and "Lab 2" correspond to which of A–E?).

`manifest.csv` lists, for every frame, the raw file it was built from, the
label map applied and its class counts.

## Sensors and recording

- Each lab is observed by two Ouster OS0-128 LiDAR sensors (128 channels);
  sensor height above the floor is 1.4–2.0 m depending on the session.
- The work cells contain Franka Emika Panda and Universal Robots UR5 arms, an
  AGV and an Industry 5.0 demonstrator: a pen factory (assembly line).
- The real scans combine two LiDARs. Their clouds were aligned with
  point-to-plane ICP (libpointmatcher), see
  [tools/registration/](../tools/registration/README.md).
- Real frames of labs A–C were cropped to the work cell and down-sampled to
  about 11 000 points (paper, Sec. 3.2). Training applies a 4.8 cm voxel grid
  to every frame.
- Simulated frames are rebuilt from the raw Isaac Sim output (full density,
  2 cm voxel grid) with the hand-corrected labels, see `tools/isaac_to_npy.py`.
  Robots have no semantic class in Isaac and were labelled by hand; this was
  done for 126 of the 209 frames of the 02.05.2023 run. The October 2022 runs
  (`sim/isaac_2022-10`) are released as converted, labelled frames of about
  58 000 points. The raw runs of 26.04.2023 (112 frames) and 27.04.2023 (25)
  still need the robot labels before they can be released.
  The later sessions (lab D, E) were recorded through the real-time node:
  crop box plus a 0.048 m voxel grid, about 5 000 points per frame.

## Labels

| id | class | description |
|----|-------|-------------|
| 0 | Robot | robot arms: Franka Emika Panda, UR5 |
| 1 | Human | people in the work cell |
| 2 | Wall | walls and fixed structures |
| 3 | Floor | ground |
| 4 | Demonstrator | Industry 5.0 pen-factory demonstrator (assembly line) |
| 5 | Table | tables and work benches |
| 6 | AGV | automated guided vehicle |
| 7 | Unlabelled | points not assigned to any class; ignored in training and evaluation |

Labelling tool: [semantic-segmentation-editor](https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor)
(v1.2.0), point-cloud mode; its PCD export is converted with `tools/pcd_to_npy.py`.

### Label order in the raw recordings

The raw recordings do not all use the same ids for AGV and Unlabelled. The
released files are remapped to the table above by `tools/build_dataset.py`
with the label maps in `tools/dataset_sources.yaml`. The raw convention of each
source was determined from the shape of the labelled points (the AGV is one
compact object, about 0.7 m wide and 1.3 m tall, standing on the floor;
Unlabelled points are scattered) and checked visually:

| raw recordings | raw 6 | raw 7 | fix |
|----------------|-------|-------|-----|
| simulation, lab B 2023-10-22 | AGV | – | none |
| lab A (`NewLabelRealDown*`), lab C (`LabDemons*`) | Unlabelled | AGV | swap 6 and 7 |
| lab B 2023-10-31 / 11-01 / 11-10, labs D and E | Unlabelled | – | 6 → 7 |

Each frame is included once.

### Label conventions found in the raw data

Five numberings were used over the course of the project. Use the label map
in the last column with `tools/pcd_to_npy.py --label-map ...` to convert data
in that numbering to the one of this dataset.

| convention | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | `--label-map` |
|---|---|---|---|---|---|---|---|---|---|
| **this dataset** (FUSION, `main2Stream.py`) | Robot | Human | Wall | Floor | Demonstrator | Table | AGV | Unlabelled | – |
| Open3D-ML `Simulation.py`; annotations from late 2023 on | Robot | Human | Wall | Floor | Demonstrator | Table | Unlabelled | AGV | `6:7 7:6` |
| annotation tool, raw export (`PCD2NPY.py`) | Unlabelled | Floor | Wall | Robot | Human | AGV | Demonstrator | Table | `0:7 1:3 3:0 4:1 5:6 6:4 7:5` |
| COVERED v1 / Open3D-ML `smartlab.py` | Unlabelled | Floor | Wall | Robot | Human | AGV | – | – | `0:7 1:3 3:0 4:1 5:6` |
| early 2022 annotations (`CustomImageDataset4Downsampling`) | Demonstrator | Floor | Wall | Robot | Human | Table | – | – | `0:4 1:3 3:0 4:1` |
| Isaac Sim raw output | per frame, see `Config <time>.txt` | | | | | | | | `tools/isaac_to_npy.py` |

The released files all use the numbering of this dataset; no remapping is
needed.

### COVERED scenarios (`real/labA_covered_s*`)

The four scenarios of the COVERED dataset (https://github.com/Fatemeh-MA/COVERED)
were recorded in lab A. COVERED labels Robot, Human, Wall, Floor and AGV
(raw numbering, see the table above). In this release the remaining points
(the demonstrator, tables and other static objects) carry the static class of
the nearest point, within 5 cm, of the labelled lab A frames of the same room;
points with no such neighbour are Unlabelled. Frames already contained in
`real/labA_NewLabelRealDown` (same scan at another resolution) are included
only once.

## Format

One file per frame: `N×4` float32 NumPy array `[x, y, z, label]`, coordinates
in metres in the sensor frame.

```python
import numpy as np
frame = np.load("data/real/session/000123.npy")
xyz, label = frame[:, :3], frame[:, 3].astype(int)
```

## Splits

`splits/*.txt` list the frames of each split of version 1.0, relative to
`data/`; the released models and results use these:

| files | data | train / val / test frames |
|-------|------|---------------------------|
| `sim_*.txt` | simulation | 90 / 6 / 30 |
| `labA_*.txt` | lab A | 35 / 14 / 26 |
| `labB_*.txt` | lab B, 4 sessions | 191 / 54 / 129 |
| `labC_*.txt` | lab C | 31 / 5 / 14 |
| `labD_*.txt` | lab D, 5 sessions | 37 / 12 / 22 |
| `labE_*.txt` | lab E | 34 / 9 / 22 |
| `real_all_train.txt`, `real_all_val.txt` | labs A–E together (the per-lab test sets stay held out) | 328 / 94 / – |

`splits/v1.1/` covers all 1252 frames of version 1.1. Every frame of version
1.0 keeps its split; a new frame of an existing session joins the split of its
most similar version 1.0 frame, and new sessions are split in blocks of
consecutive frames (`tools/extend_splits.py`):

| files | train / val / test frames |
|-------|---------------------------|
| `sim_*.txt` | 180 / 16 / 50 |
| `labA_*.txt` (incl. COVERED) | 123 / 35 / 60 |
| `labB_*.txt` | 225 / 70 / 164 |
| `labC_*.txt` | 31 / 5 / 14 |
| `labD_*.txt` | 38 / 12 / 23 |
| `labE_*.txt` | 37 / 9 / 22 |
| `demo_*.txt` | 98 / 15 / 25 |
| `real_all_train.txt`, `real_all_val.txt` | 552 / 146 / – |

Consecutive frames are nearly identical, so frames are split in blocks of
consecutive frames (10 for simulation, 5 for the real sessions), never one by
one; each real session is split on its own so that it appears in every split.
`tools/make_splits.py` refuses any split in which a frame and a copy of it end
up on different sides. Augmentation is done on the fly during training and is
not part of the dataset.

## Privacy

The scans contain people. Point clouds at this resolution do not show faces or
other identifying details. **TODO**: confirm that participants consented to
publication and add the approval reference, if any.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the
paper (see the main README) when you use the data.
