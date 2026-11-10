# Dataset card

> Items marked **TODO** need to be filled in by the authors before release.

## Summary

LiDAR scans of human-robot collaboration workspaces with per-point semantic
labels: 761 frames, 13 recording sessions, two domains.

| folder | domain | frames | used for |
|--------|--------|--------|----------|
| `sim/isaac_2023-05-02` | NVIDIA Isaac Sim (TODO: version) | 126 | pre-training |
| `real/labA_NewLabelRealDown` | lab A, 2023 | 75 | fine-tuning / test |
| `real/labB_2023-10-22`, `_2023-10-31`, `_2023-11-01`, `_2023-11-10` | lab B | 42 + 135 + 101 + 96 | fine-tuning / test |
| `real/labC_2023-11-20` | lab C, with demonstrator | 50 | fine-tuning / test |
| `real/labD_2023-11-29` … `_2024-06-07` (5 sessions) | lab D | 71 | fine-tuning / test |
| `real/labE_2026-08-27` | lab E | 65 | fine-tuning / test (recorded after the paper) |

TODO: lab names (the paper's "Lab 1" and "Lab 2" correspond to which of A–E?).

`manifest.csv` lists, for every frame, the raw file it was built from, the
label map applied and its class counts.

## Sensors and recording

- Each lab is observed by two Ouster LiDAR sensors (TODO: model and channels);
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
  done for 126 of the 209 frames of the 02.05.2023 run. The other raw runs
  (26.04.2023: 112 frames, 27.04.2023: 25, and the 2022 runs) still need the
  robot labels before they can be released.
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

The original COVERED dataset (https://github.com/Fatemeh-MA/COVERED, 4
scenarios, 5 classes) is not part of this release.

## Format

One file per frame: `N×4` float32 NumPy array `[x, y, z, label]`, coordinates
in metres in the sensor frame.

```python
import numpy as np
frame = np.load("data/real/session/000123.npy")
xyz, label = frame[:, :3], frame[:, 3].astype(int)
```

## Splits

`splits/*.txt` list the frames of each split, relative to `data/`:

| files | data | train / val / test frames |
|-------|------|---------------------------|
| `sim_*.txt` | simulation | 90 / 6 / 30 |
| `labA_*.txt` | lab A | 35 / 14 / 26 |
| `labB_*.txt` | lab B, 4 sessions | 191 / 54 / 129 |
| `labC_*.txt` | lab C | 31 / 5 / 14 |
| `labD_*.txt` | lab D, 5 sessions | 37 / 12 / 22 |
| `labE_*.txt` | lab E | 34 / 9 / 22 |
| `real_all_train.txt`, `real_all_val.txt` | labs A–E together (the per-lab test sets stay held out) | 328 / 94 / – |

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
