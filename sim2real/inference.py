"""Whole-frame prediction and checkpoint I/O."""

import os

import numpy as np
import torch

from .data import Normalizer, load_frame, nearest_chunk
from .metrics import ConfusionMatrix
from .models import build_model


@torch.no_grad()
def predict_frame(model, points, num_points, normalizer, device, seed=0,
                  exclude_class=None, floor=None):
    """Predict a label for EVERY point of a frame.

    A frame with at most `num_points` points goes through the network in one
    pass. A larger frame is covered with contiguous chunks of `num_points`
    neighbouring points, so the density stays the same as in training.

    points: (N, 3) array in metres, sensor frame. `floor`: optional cached
    floor fit, see `Normalizer.align`. Returns an (N,) int64 array.
    """
    n = len(points)
    if n <= num_points:
        x = torch.from_numpy(normalizer(points, floor)).to(device)
        logits = model(x.t().unsqueeze(0))[0]
        if exclude_class is not None:
            logits[exclude_class] = float("-inf")
        return logits.argmax(0).cpu().numpy()

    # larger frame: cover it with contiguous chunks of `num_points` points,
    # each centred on itself like a training piece
    rng = np.random.default_rng(seed)
    aligned = normalizer.align(points, floor)
    pred = np.full(n, -1, dtype=np.int64)
    while (pred < 0).any():
        seed_index = rng.choice(np.flatnonzero(pred < 0))
        chunk = nearest_chunk(points, seed_index, num_points)
        x = torch.from_numpy(normalizer.finish(normalizer.recenter(aligned[chunk]))).to(device)
        logits = model(x.t().unsqueeze(0))[0]
        if exclude_class is not None:
            logits[exclude_class] = float("-inf")
        new = pred[chunk] < 0                     # first prediction of a point is kept
        pred[chunk[new]] = logits.argmax(0).cpu().numpy()[new]
    return pred


def untrained_class(ckpt):
    """Class whose logit was never trained (loss ignore_index), or None.

    Its logit is excluded at prediction time. Original checkpoints trained all
    eight outputs.
    """
    cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    return (cfg or {}).get("train", {}).get("ignore_index")


def evaluate_paths(model, paths, num_points, normalizer, device, num_classes,
                   ignore_index=None, on_frame=None, exclude_class=None):
    """Whole-frame evaluation of `paths`. Returns a filled ConfusionMatrix.

    Points labelled `ignore_index` are not scored. `exclude_class` is never
    predicted. `on_frame(path, points, labels, pred)` is called after each
    frame, e.g. to save predictions.
    """
    was_training = model.training
    model.eval()
    cm = ConfusionMatrix(num_classes, ignore_index)
    for i, path in enumerate(paths):
        points, labels = normalizer.downsample(*load_frame(path))
        pred = predict_frame(model, points, num_points, normalizer, device, seed=i,
                             exclude_class=exclude_class)
        cm.update(pred, labels)
        if on_frame is not None:
            on_frame(path, points, labels, pred)
    model.train(was_training)
    return cm


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_checkpoint(path, model, normalizer, config, epoch, optimizer=None,
                    scheduler=None, metrics=None):
    """Write atomically, so a reader or a crash never sees a half-written file."""
    tmp = path + ".tmp"
    torch.save({
        "model": unwrap(model).state_dict(),
        "model_name": config["model"]["name"],
        "model_args": config["model"].get("args", {}),
        "normalization": normalizer.to_dict(),
        "config": config,
        "epoch": epoch,
        "metrics": metrics or {},
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu", legacy_model="dualstream"):
    """Load a checkpoint from this repo OR from the original training code.

    Original checkpoints ({'model_state_dict': ...} saved through
    nn.DataParallel) carry no normalization: those models were trained on raw
    coordinates in metres, so the identity normalizer is returned for them.

    Returns (model, normalizer, checkpoint_dict).
    """
    ckpt = torch.load(path, map_location=device)
    if "model" in ckpt and "normalization" in ckpt:
        state = ckpt["model"]
        name, args = ckpt["model_name"], dict(ckpt["model_args"])
        normalizer = Normalizer.from_dict(ckpt["normalization"])
    else:  # legacy format
        state = ckpt.get("model_state_dict", ckpt)
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}
        name = legacy_model
        last = getattr(build_model(name), "LAST")
        args = {"num_classes": state[last + ".weight"].shape[0]}
        normalizer = Normalizer.identity()
    model = build_model(name, **args)
    model.load_state_dict(state)
    return model.to(device), normalizer, ckpt
