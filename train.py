#!/usr/bin/env python3
"""Train (or fine-tune) a segmentation model.

    python train.py --config configs/sim_pretrain.yaml
    python train.py --config configs/finetune.yaml --set finetune.init_from=outputs/sim/best.pth
    python train.py --config configs/sim_pretrain.yaml --set train.epochs=5 output_dir=outputs/debug

    # continue a stopped run, optionally with a new learning rate
    python train.py --config outputs/sim/config.yaml --resume outputs/sim/last.pth --lr 1e-4
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from sim2real.config import load_config, seed_everything
from sim2real.data import Normalizer, PointCloudDataset, read_split, worker_init_fn
from sim2real.inference import evaluate_paths, load_checkpoint, save_checkpoint, unwrap
from sim2real.labels import CLASS_NAMES, UNLABELLED
from sim2real.models import build_model


def build_optimizer(params, cfg):
    name = cfg["name"].lower()
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9),
                               weight_decay=cfg["weight_decay"])
    raise ValueError("unknown optimizer %r" % name)


def build_scheduler(optimizer, cfg, epochs):
    """cosine: decay to `min_lr` over the run (the paper).
    step: multiply by `gamma` every `step_size` epochs, never below `min_lr`."""
    base = optimizer.param_groups[0]["lr"]
    floor = cfg.get("min_lr", 0.0) / base
    if cfg.get("name", "step") == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=cfg.get("min_lr", 0.0))

    def factor(epoch):
        return max(cfg["gamma"] ** (epoch // cfg["step_size"]), floor)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def setup_model(cfg, train_paths, device):
    """Build the model and its normalizer, from scratch or from a checkpoint."""
    ft = cfg.get("finetune")
    if ft:
        model, normalizer, _ = load_checkpoint(ft["init_from"], device)
        print("initialised from %s, %r" % (ft["init_from"], normalizer))
        if ft.get("reset_head"):
            model.reset_head(cfg["model"]["args"]["num_classes"])
        if ft.get("freeze_encoders"):        # train only the last layers, as in the paper
            for name, p in model.named_parameters():
                p.requires_grad = name.split(".")[0] in model.HEAD
        return model, normalizer

    model = build_model(cfg["model"]["name"], **cfg["model"].get("args", {})).to(device)
    normalizer = Normalizer.fit(train_paths, cfg.get("normalization", "floor"),
                                voxel=cfg["data"].get("voxel_size"))
    print(repr(normalizer))
    return model, normalizer


def freeze_batchnorm(model):
    """Put the BatchNorm layers outside the head into eval mode.

    Frozen layers then keep their running statistics; with a few frames and
    batch size 1 they would otherwise be overwritten by the new data.
    """
    model = unwrap(model)
    for name, module in model.named_modules():
        if name and name.split(".")[0] not in model.HEAD \
                and isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def format_scores(cm):
    s = cm.summary()
    return "mIoU %.4f  OA %.4f  mAcc %.4f" % (s["mIoU"], s["OA"], s["mAcc"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, e.g. train.epochs=5")
    ap.add_argument("--resume", metavar="LAST_PTH",
                    help="continue from a checkpoint of this run (model, optimizer, scheduler, epoch)")
    ap.add_argument("--lr", type=float,
                    help="with --resume: new learning rate from here on (the schedule continues from it)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tcfg, dcfg = cfg["train"], cfg["data"]
    seed_everything(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    train_paths = read_split(dcfg["train_split"], dcfg["root"])
    val_paths = read_split(dcfg["val_split"], dcfg["root"]) if dcfg.get("val_split") else []
    print("train frames: %d, val frames: %d" % (len(train_paths), len(val_paths)))

    model, normalizer = setup_model(cfg, train_paths, device)
    normalizer.save(os.path.join(out, "normalization.json"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable parameters: {:,}".format(n_train))

    # batch size 1: small frames are used whole instead of padded with repeats
    dataset = PointCloudDataset(train_paths, dcfg["num_points"], normalizer,
                                augmentation=cfg.get("augmentation"), seed=tcfg["seed"],
                                pad=tcfg["batch_size"] > 1)
    loader = DataLoader(dataset, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=tcfg["num_workers"], worker_init_fn=worker_init_fn,
                        pin_memory=device.type == "cuda")

    num_classes = cfg["model"]["args"]["num_classes"]
    ignore = tcfg.get("ignore_index")                          # loss; None = train every class
    score_ignore = tcfg.get("score_ignore_index", UNLABELLED)  # not counted in the metrics
    weights = tcfg.get("class_weights")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device) if weights else None,
        ignore_index=-100 if ignore is None else ignore)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(params, tcfg["optimizer"])
    scheduler = build_scheduler(optimizer, tcfg["scheduler"], tcfg["epochs"])
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("using %d GPUs" % torch.cuda.device_count())

    best, start = -1.0, 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        unwrap(model).load_state_dict(ckpt["model"])
        normalizer = Normalizer.from_dict(ckpt["normalization"])
        dataset.normalizer = normalizer
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start = ckpt["epoch"] + 1
        if os.path.exists(os.path.join(out, "best.pth")):
            best = torch.load(os.path.join(out, "best.pth"), map_location="cpu")["metrics"].get("mIoU", -1.0)
        if args.lr:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
            scheduler.base_lrs = [args.lr] * len(scheduler.base_lrs)
        print("resumed from %s at epoch %d, lr %.2e" % (args.resume, start, optimizer.param_groups[0]["lr"]))

    log_path = os.path.join(out, "log.csv")
    if not args.resume or not os.path.exists(log_path):
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch", "lr", "train_loss", "train_acc", "val_mIoU",
                                     "val_OA", "val_mAcc", "seconds"])

    ft = cfg.get("finetune") or {}
    for epoch in range(start, tcfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        if ft.get("freeze_bn"):
            freeze_batchnorm(model)
        total_loss, seen, correct, counted = 0.0, 0, 0, 0
        for points, labels in loader:
            points = points.to(device).transpose(1, 2)        # (B, 3, N)
            labels = labels.to(device)
            logits = model(points)                            # (B, C, N)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            if tcfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
            optimizer.step()

            total_loss += loss.item() * len(points)
            seen += len(points)
            valid = labels != score_ignore
            correct += (logits.argmax(1)[valid] == labels[valid]).sum().item()
            counted += valid.sum().item()
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        train_loss = total_loss / max(seen, 1)
        train_acc = correct / max(counted, 1)

        val = {"mIoU": float("nan"), "OA": float("nan"), "mAcc": float("nan")}
        if val_paths and epoch % tcfg.get("val_every", 1) == 0:
            cm = evaluate_paths(unwrap(model), val_paths, dcfg["num_points"], normalizer,
                                device, num_classes, score_ignore, exclude_class=ignore)
            val = cm.summary()

        seconds = time.time() - t0
        print("epoch %3d  lr %.2e  loss %.4f  acc %.4f  val mIoU %.4f  (%.0fs)"
              % (epoch, lr, train_loss, train_acc, val["mIoU"], seconds))
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, lr, train_loss, train_acc, val["mIoU"], val["OA"],
                                     val["mAcc"], round(seconds, 1)])

        save_checkpoint(os.path.join(out, "last.pth"), model, normalizer, cfg, epoch,
                        optimizer, scheduler, val)
        if not np.isnan(val["mIoU"]) and val["mIoU"] > best:
            best = val["mIoU"]
            save_checkpoint(os.path.join(out, "best.pth"), model, normalizer, cfg, epoch,
                            metrics=val)

    if not os.path.exists(os.path.join(out, "best.pth")):   # no validation split
        save_checkpoint(os.path.join(out, "best.pth"), model, normalizer, cfg, tcfg["epochs"])

    if dcfg.get("test_split"):
        model, normalizer, _ = load_checkpoint(os.path.join(out, "best.pth"), device)
        cm = evaluate_paths(model, read_split(dcfg["test_split"], dcfg["root"]),
                            dcfg["num_points"], normalizer, device, num_classes, score_ignore,
                            exclude_class=ignore)
        print("\ntest (best checkpoint): " + format_scores(cm))
        for name, iou in zip(CLASS_NAMES, cm.iou()):
            print("  %-13s %s" % (name, "-" if np.isnan(iou) else "%.4f" % iou))


if __name__ == "__main__":
    main()
