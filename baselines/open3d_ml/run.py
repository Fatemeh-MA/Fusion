#!/usr/bin/env python3
"""RandLA-Net / KPConv baselines (paper, Table 1) with Open3D-ML on the same splits.

    # train on simulation, then fine-tune and test on a real lab
    python baselines/open3d_ml/run.py train --config baselines/open3d_ml/randlanet.yml \
        --train splits/sim_train.txt --val splits/sim_val.txt --out outputs/randlanet_sim
    python baselines/open3d_ml/run.py train --config baselines/open3d_ml/randlanet.yml \
        --train splits/labD_train.txt --val splits/labD_val.txt --out outputs/randlanet_labD \
        --ckpt outputs/randlanet_sim/checkpoint/<latest>.pth --finetune
    python baselines/open3d_ml/run.py test --config baselines/open3d_ml/randlanet.yml \
        --test splits/labD_test.txt --ckpt outputs/randlanet_labD/checkpoint/<latest>.pth

Needs Open3D with its PyTorch ML ops (Open3D 0.15 + PyTorch 1.8 here). The
frames get the same floor alignment as FUSION (in metres, since these networks
sub-sample on a metric grid), so results are comparable.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from sim2real.data import Normalizer, estimate_floor, load_frame, read_split  # noqa: E402
from sim2real.labels import CLASS_NAMES, UNLABELLED  # noqa: E402
from sim2real.metrics import ConfusionMatrix  # noqa: E402

import open3d.ml as _ml3d  # noqa: E402
import open3d.ml.torch as ml3d  # noqa: E402
try:  # noqa: E402
    from open3d._ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit
except ImportError:  # external Open3D-ML (OPEN3D_ML_ROOT)
    from ml3d.datasets.base_dataset import BaseDataset, BaseDatasetSplit


# Open3D-ML 0.15 computes a metric inside run_inference with tensors on mixed
# devices; that metric is not used here, so compute it on the CPU.
import open3d._ml3d.torch.pipelines.semantic_segmentation as _semseg  # noqa: E402
_filter_valid_label = _semseg.filter_valid_label
_semseg.filter_valid_label = lambda s, l, n, ign, device: _filter_valid_label(s, l, n, ign, "cpu")


class SplitFileDataset(BaseDataset):
    """Open3D-ML dataset reading this repository's split files."""

    def __init__(self, splits, data_root="data", name="Sim2Real", **kwargs):
        super().__init__(dataset_path=data_root, name=name, **kwargs)
        self.splits = {k: (read_split(v, data_root) if v else []) for k, v in splits.items()}
        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.aligner = Normalizer("floor")

    @staticmethod
    def get_label_to_names():
        return dict(enumerate(CLASS_NAMES))

    def get_split(self, split):
        return SplitFileDatasetSplit(self, split)

    def get_split_list(self, split):
        key = {"training": "train", "train": "train", "validation": "val", "val": "val",
               "test": "test", "testing": "test"}[split]
        return self.splits[key]

    def is_tested(self, attr):
        return False

    def save_test_result(self, results, attr):
        pass


class SplitFileDatasetSplit(BaseDatasetSplit):
    def __init__(self, dataset, split="training"):
        super().__init__(dataset, split=split)
        self.path_list = dataset.get_split_list(split)

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        points, labels = load_frame(self.path_list[idx])
        points = self.dataset.aligner.align(points, estimate_floor(points))
        return {"point": points.astype(np.float32), "feat": None, "label": labels.astype(np.int32)}

    def get_attr(self, idx):
        name = os.path.splitext(os.path.basename(self.path_list[idx]))[0]
        return {"name": name, "path": self.path_list[idx], "split": self.split}


def build(cfg, dataset, out):
    model = ml3d.models.RandLANet(**cfg.model) if cfg.model["name"] == "RandLANet" \
        else ml3d.models.KPFCNN(**cfg.model)
    cfg.pipeline["main_log_dir"] = out
    cfg.pipeline["train_sum_dir"] = os.path.join(out, "tensorboard")
    pipeline = ml3d.pipelines.SemanticSegmentation(model=model, dataset=dataset, **cfg.pipeline)
    return model, pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "test"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--train")
    ap.add_argument("--val")
    ap.add_argument("--test")
    ap.add_argument("--ckpt", help="checkpoint to start from (train) or to evaluate (test)")
    ap.add_argument("--finetune", action="store_true",
                    help="train only the decoder and the last layers, like FUSION's head")
    ap.add_argument("--out", default="outputs/open3d_ml")
    a = ap.parse_args()

    cfg = _ml3d.utils.Config.load_from_file(a.config)
    dataset = SplitFileDataset({"train": a.train, "val": a.val, "test": a.test}, a.data_root,
                               **cfg.dataset)
    model, pipeline = build(cfg, dataset, a.out)
    if a.ckpt:
        pipeline.load_ckpt(ckpt_path=a.ckpt)

    if a.mode == "train":
        if a.finetune:
            for name, p in model.named_parameters():
                p.requires_grad = name.startswith(("decoder", "fc", "head"))
        pipeline.run_train()
        return

    cm = ConfusionMatrix(len(CLASS_NAMES), UNLABELLED)
    split = dataset.get_split("test")
    for i in range(len(split)):
        data = split.get_data(i)
        pred = pipeline.run_inference(data)["predict_labels"].astype(np.int64)
        cm.update(pred, data["label"])
    s = cm.summary()
    print("\n%d frames  mIoU %.4f  OA %.4f  mAcc %.4f" % (len(split), s["mIoU"], s["OA"], s["mAcc"]))
    for name, iou in zip(CLASS_NAMES, cm.iou()):
        print("  %-13s %s" % (name, "-" if np.isnan(iou) else "%.4f" % iou))


if __name__ == "__main__":
    main()
