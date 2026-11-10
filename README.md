# FUSION: Sim2Real Point Cloud Segmentation for Human-Robot Collaboration

Code and dataset for

> **Enhancing Human-Robot Collaboration: A Sim2Real Domain Adaptation Algorithm for
> Point Cloud Segmentation in Industrial Environments**
> Fatemeh Mohammadi Amin, Darwin G. Caldwell, Hans Wernher van de Venn.
> *Journal of Intelligent & Robotic Systems* 111:94, 2025.
> [doi:10.1007/s10846-025-02290-9](https://doi.org/10.1007/s10846-025-02290-9)

FUSION is a dual-stream network that segments LiDAR point clouds of
human-robot collaboration cells. It is trained on simulated scans (NVIDIA
Isaac Sim) and adapted to a real cell by fine-tuning only its segmentation
head on a few labelled real frames.

```
            ┌─ CNN-residual stream (per-point) ─┐
xyz (N×3) ──┤                                   ├─ concat ─ conv8 ─ conv9 ─ scores (N×8)
            └─ DGCNN stream (EdgeConv, k = 40) ─┘
```

## Classes

| id | class | id | class |
|----|-------|----|-------|
| 0 | Robot | 4 | Demonstrator (assembly line) |
| 1 | Human | 5 | Table |
| 2 | Wall | 6 | AGV |
| 3 | Floor | 7 | Unlabelled (trained, but not scored) |

The single source of truth is [sim2real/labels.py](sim2real/labels.py).

## Installation

Tested with Python 3.7, PyTorch 1.8.1 and CUDA 11.1.

```bash
git clone https://github.com/Fatemeh-MA/Fusion.git
cd Fusion
conda env create -f environment.yml
conda activate sim2real
python -m unittest discover tests        # ~6 s on CPU
```

## Data

Download the dataset from TODO and extract it into `data/`:

```
data/
├── sim/isaac_2023-05-02/*.npy     simulated frames (NVIDIA Isaac Sim)
├── real/labA_…/ … real/labE_…/    real frames, one folder per recording session
└── manifest.csv                   source and label fix of every frame
```

Each frame is an `N×4` float32 array `[x, y, z, label]` in metres, in the
frame of the (merged) LiDAR sensors. [docs/DATASET.md](docs/DATASET.md)
describes the sessions, the labels and how the files were built.

Split files are in [splits/](splits/): `sim_*` for pre-training and
`labA_*` … `labE_*` for fine-tuning and testing in each real environment.
Frames are split in blocks of consecutive frames so that no test frame has a
near-identical neighbour in training; for the real labs every recording
session is split on its own (`--per-folder`), so each session is in train,
validation and test:

```bash
python tools/make_splits.py --root data --only sim --block-size 10 --val 0.1 --test 0.2 --prefix sim_
python tools/make_splits.py --root data --only real/labD_2023-11-29 real/labD_2023-12-05 \
    real/labD_2023-12-06 real/labD_2024-05-03 real/labD_2024-06-07 \
    --block-size 5 --val 0.15 --test 0.35 --per-folder --prefix labD_
```

## Training

As in the paper: train on simulation, then fine-tune on the real environment.

```bash
# 1. pre-train on simulated frames
python train.py --config configs/sim_pretrain.yaml

# 2. fine-tune on a few labelled frames of a real environment
python train.py --config configs/finetune.yaml \
    --set finetune.init_from=outputs/sim/best.pth output_dir=outputs/finetune_labD

# 3. a new environment starts from the latest fine-tuned model
python train.py --config configs/finetune.yaml \
    --set finetune.init_from=outputs/finetune_labD/best.pth \
          data.train_split=splits/labE_train.txt data.val_split=splits/labE_val.txt \
          data.test_split=splits/labE_test.txt output_dir=outputs/finetune_labE
```

Each run writes `config.yaml`, `normalization.json`, `log.csv` (one line per
epoch, learning rate included), `last.pth` and `best.pth` (best validation
mIoU) to its `output_dir`. Any config entry can be overridden with
`--set key.sub=value`. A stopped run continues with
`--resume outputs/sim/last.pth`, optionally with a new learning rate (`--lr`).

### Training recipe

| | pre-training on simulation | fine-tuning on a real lab |
|---|---|---|
| optimizer | Adam, lr 1e-3, weight decay 1e-4 (paper, Sec. 4.1) | Adam, lr 1e-4, weight decay 5.6e-4, gradient clip 10 |
| schedule | cosine to 1e-5 over 100 epochs | cosine to 1e-5 over 50 epochs |
| batch | 10 pieces of ≤ 10 500 points | 1 frame, used whole |
| trained layers | all | all (the paper trained conv8/conv9 only; see `configs/finetune.yaml`) |
| loss | cross-entropy, class weights [0.14 0.15 0.12 0.15 0.12 0.12 0.16 0.07] | same |

The original script used RMSprop (lr 1.76e-4, weight decay 6.76e-4, a 30x
step at epoch 15, found by a hyper-parameter search in 2023). It is still
available:

```bash
python train.py --config configs/sim_pretrain.yaml \
    --set "train.optimizer={name: rmsprop, lr: 1.76e-4, weight_decay: 6.76e-4}" \
          "train.scheduler={name: step, step_size: 15, gamma: 0.03143, min_lr: 1.0e-5}"
```

### Normalization: independent of the lab, the sensor pose and the data source

Every frame is normalized on its own, without a reference cloud. The
parameters are stored in the checkpoint and reused by evaluation and the
real-time node:

1. **density**: a 4.8 cm voxel grid, the one of the real-time node, so a frame
   has the same point density whether it is simulated, recorded or live;
2. **floor**: the floor plane is fitted (RANSAC, no labels needed), levelled
   and put at z = 0;
3. **position**: the median x/y of the frame is moved to the origin;
4. **scale**: one fixed scale learnt on the training split (never per frame).

The network therefore sees *height above the floor*, positions relative to
the scene and a fixed density, which mean the same thing in simulation, in
every lab and after the sensors are moved. Training adds random rotations,
shifts and crop boxes, and feeds contiguous pieces of large scenes instead of
random subsets. `tools/eval_sensor_moves.py` re-evaluates a test split as if
the sensor had been mounted elsewhere.

## Evaluation

```bash
python evaluate.py --ckpt outputs/finetune_labD/best.pth --split splits/labD_test.txt
python evaluate.py --ckpt outputs/finetune_labD/best.pth --split splits/labD_test.txt \
    --save-predictions outputs/pred --out outputs/labD_metrics.json
python tools/visualize.py outputs/pred/*.npy --pred     # N/P: next/previous, L: label/prediction
python tools/eval_sensor_moves.py --ckpt outputs/finetune_labD/best.pth --split splits/labD_test.txt
```

Every frame is scored on its 4.8 cm voxel grid (the density the network
sees), every voxel included; IoU is accumulated over the whole split.
Unlabelled points are not scored, and classes with fewer than 100 points in
a split are left out of the means (`--min-points`).

## Tools

| script | purpose |
|--------|---------|
| `tools/pcd_to_npy.py` | labelled `.pcd` (ascii/binary) → `[x, y, z, label]` `.npy`, with `--label-map` |
| `tools/npy_to_pcd.py` | `.npy` frames or predictions → `.pcd` for (re-)annotation |
| `tools/crop_frames.py` | cut the work cell out of frames with a box (optionally rotated) |
| `tools/visualize.py` | browse `.npy`/`.pcd` frames and predictions (`--browser`: Open3D-ML viewer) |
| `tools/isaac_to_npy.py` | rebuild full-density simulated frames from raw Isaac Sim output |
| `tools/build_dataset.py` | assemble the released dataset from the raw recordings (`tools/dataset_sources.yaml`) |
| `tools/make_splits.py` | train/val/test splits in blocks of consecutive frames |
| `tools/eval_sensor_moves.py` | accuracy when the sensor is moved, raised, tilted or turned |
| `tools/registration/` | ICP registration of the two LiDARs (libpointmatcher) |

## Baselines

The single-stream networks of Table 1 train with the same scripts:

```bash
python train.py --config configs/sim_pretrain.yaml --set model.name=dgcnn output_dir=outputs/dgcnn_sim
# model.name: dualstream (FUSION), pointnet, dgcnn, cnn_residual
```

RandLA-Net and KPConv use Open3D-ML on the same splits and normalization:

```bash
python baselines/open3d_ml/run.py train --config baselines/open3d_ml/randlanet.yml \
    --train splits/sim_train.txt --val splits/sim_val.txt --out outputs/randlanet_sim
python baselines/open3d_ml/run.py test --config baselines/open3d_ml/randlanet.yml \
    --test splits/labD_test.txt --ckpt outputs/randlanet_sim/RandLANet_Sim2Real_torch/checkpoint/ckpt_00100.pth
```

Checkpoints of the original code load with
`load_checkpoint(path, legacy_model="dgcnn" | "pointnet" | "cnn_residual")`.

## Real-time node (ROS 1)

```bash
# edit the checkpoint and crop box in configs/realtime.yaml, then
python3 realtime/segmentation_node.py --config configs/realtime.yaml
# after moving the sensors:
rosservice call /reset_floor
# without sensors: replay recorded frames on /merged/cloud
python3 realtime/replay_frames.py data/real/labE_2026-08-27/*.npy --rate 10 --loop
```

The node subscribes to the merged cloud of both LiDARs (`/merged/cloud`),
crops and voxel-downsamples it like the recorded frames, and publishes
`/segmented_point_cloud` (dynamic classes, coloured), `/segmented_point_cloud_label`
and, once, `/Wall`, `/floor` and `/Table`. The two LiDARs are registered with
ICP beforehand, see [tools/registration/](tools/registration/README.md).
It needs `rospy` and PyTorch in the same Python environment.

## Pretrained models

| checkpoint | trained on | use |
|------------|-----------|-----|
| `fusion_sim.pth` | simulated frames | starting point for fine-tuning |
| `fusion_real.pth` | simulation, then real frames of labs A–E | **recommended**: real-time use and fine-tuning in a new environment |
| `fusion_labD.pth` | simulation, then lab D | lab D |
| `fusion_labE.pth` | simulation, lab D, then lab E | lab E |
| `fusion_original_2024-06-07.pth` | original training code of the paper, lab D fine-tune of 7 June 2024 | the deployed lab-D model |
| `fusion_original_2026-09-16.pth` | original training code, lab E fine-tune of 16 September 2026 | the deployed lab-E model |

Download: TODO (GitHub release).

The two `fusion_original_*` checkpoints are the models of the original code,
converted to this format (raw sensor coordinates, 4.8 cm voxel grid; outputs
6 and 7 in the order of `sim2real/labels.py`). They expect the sensor pose and
crop box they were trained with.

Results on held-out test frames, never used for training or model selection.
`python evaluate.py --ckpt <checkpoint> --split splits/<set>_test.txt`
reproduces every number:

| test set | checkpoint | overall accuracy | mean class accuracy | mIoU |
|----------|------------|------------------|---------------------|------|
| simulation | `fusion_sim.pth` | 96.6 % | 82.2 % | 0.744 |
| lab A | `fusion_real.pth` | 93.5 % | 88.5 % | 0.816 |
| lab B | `fusion_real.pth` | 96.2 % | 82.5 % | 0.766 |
| lab C | `fusion_real.pth` | 95.8 % | 74.8 % | 0.694 |
| lab D | `fusion_real.pth` | 89.6 % | 78.4 % | 0.688 |
| lab D | `fusion_labD.pth` | 87.7 % | 76.1 % | 0.657 |
| lab E | `fusion_real.pth` | 92.2 % | 86.7 % | 0.806 |
| lab E | `fusion_labE.pth` | 94.7 % | 92.7 % | 0.839 |

Means are over the classes with at least 100 test points; Unlabelled is not
scored. The results stay the same when the sensor is moved, raised, tilted or
turned (`tools/eval_sensor_moves.py`).

Checkpoints store their normalization, so `evaluate.py` and the real-time
node need nothing else. Checkpoints of the original code (raw coordinates)
load too.

## A new environment

Model-assisted labelling, then a short fine-tune:

```bash
# 1. record frames with the real-time node (or replay them) and predict
python evaluate.py --ckpt fusion_real.pth --split splits/new.txt --save-predictions pred/
# 2. predictions -> PCD, correct them in the annotation tool, convert back
python tools/npy_to_pcd.py pred/ --out to_fix/ --use-prediction
python tools/pcd_to_npy.py to_fix/ --out data/real/labF_<date>/
# 3. split in blocks of consecutive frames, fine-tune, test on held-out frames
python tools/make_splits.py --root data --only real/labF_<date> --block-size 5 --per-folder --prefix labF_
python train.py --config configs/finetune.yaml --set finetune.init_from=fusion_real.pth \
    data.train_split=splits/labF_train.txt data.val_split=splits/labF_val.txt \
    data.test_split=splits/labF_test.txt output_dir=outputs/finetune_labF
```

About 15–35 corrected frames are enough; start from `fusion_real.pth` and
keep the checkpoint of each environment. Choose them from different moments
(people, robot poses and AGV in different places), check the corrections
carefully where people and robots touch, and judge the result only on
frames from another part of the recording: consecutive frames are nearly
identical. Keep the label numbers of `sim2real/labels.py` (6 = AGV,
7 = Unlabelled).

## Changes compared with the code of the paper

The network is the one of the paper; checkpoints of the original code load
unchanged. This release adds:

- **Normalization independent of the installation**: voxel grid, floor
  alignment, per-piece centring and a fixed scale, stored in the checkpoint.
  The same model works when the sensors are moved or the crop box changes.
- **One class numbering for all recordings**; the raw recordings used two
  orders for AGV and Unlabelled (see [docs/DATASET.md](docs/DATASET.md)).
- **Uniform point density**: simulated frames rebuilt at full density from
  the raw Isaac Sim output.
- **Online augmentation** (rotation, flip, scale, shift, crop box, jitter).
- **Splits** in blocks of consecutive frames, per recording session.
- **Whole-network fine-tuning** at a low learning rate, which gave the best
  results; the head-only fine-tuning of the paper is a config switch away.
- **Unlabelled** is learnt as a class but not scored.

Use `--set normalization=none` to train on raw coordinates like the paper.

## Citation

```bibtex
@article{mohammadiamin2025fusion,
  title   = {Enhancing Human-Robot Collaboration: A Sim2Real Domain Adaptation Algorithm
             for Point Cloud Segmentation in Industrial Environments},
  author  = {Mohammadi Amin, Fatemeh and Caldwell, Darwin G. and van de Venn, Hans Wernher},
  journal = {Journal of Intelligent \& Robotic Systems},
  volume  = {111},
  pages   = {94},
  year    = {2025},
  doi     = {10.1007/s10846-025-02290-9}
}
```

The real-world recordings relate to the COVERED dataset
([github.com/Fatemeh-MA/COVERED](https://github.com/Fatemeh-MA/COVERED)).

## License

Code: [MIT](LICENSE). Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The DGCNN stream follows *Dynamic Graph CNN for Learning on Point Clouds*
(Wang et al., ACM TOG 2019).
