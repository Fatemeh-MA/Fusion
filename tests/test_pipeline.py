"""Fast CPU checks. Run with:  python -m unittest discover tests"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim2real.data import (Normalizer, PointCloudDataset, augment, estimate_floor,  # noqa: E402
                           nearest_chunk, voxel_downsample)
from sim2real.io import read_pcd, write_pcd  # noqa: E402
from sim2real.models import build_model, output_layer  # noqa: E402
import train as train_script  # noqa: E402
from sim2real.inference import load_checkpoint, predict_frame, save_checkpoint  # noqa: E402
from sim2real.metrics import ConfusionMatrix  # noqa: E402
from sim2real.models import DualStreamNetwork  # noqa: E402


def write_frames(folder, n_frames, rng, n_points=300):
    os.makedirs(folder, exist_ok=True)
    for i in range(n_frames):
        pts = rng.normal(0, 3, (n_points, 3)).astype(np.float32)
        lab = rng.integers(0, 8, n_points).astype(np.float32)
        np.save(os.path.join(folder, "frame_%03d.npy" % i), np.column_stack([pts, lab]))


class ModelTest(unittest.TestCase):
    def test_forward_shape(self):
        model = DualStreamNetwork(num_classes=8, k=10).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 64))
        self.assertEqual(tuple(out.shape), (2, 8, 64))

    def test_reset_head(self):
        model = DualStreamNetwork(num_classes=8, k=10)
        model.reset_head(5)
        self.assertEqual(model.conv9.out_channels, 5)


def scene(rng, n_floor=3000):
    """Floor grid at z = 0 plus two boxes, in metres."""
    floor = np.c_[rng.uniform(-3, 3, (n_floor, 2)), np.zeros(n_floor)]
    box1 = rng.uniform([0.5, 0.5, 0.0], [1.0, 1.0, 1.7], (500, 3))
    box2 = rng.uniform([-2.0, -1.0, 0.0], [-1.2, 0.0, 0.8], (500, 3))
    return np.vstack([floor, box1, box2])


def move_sensor(points, yaw_deg, tilt_deg, translation):
    """Express a scene in the frame of a sensor with a different pose."""
    y, t = np.radians(yaw_deg), np.radians(tilt_deg)
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return points @ (rx @ rz).T + translation


class FloorAlignmentTest(unittest.TestCase):
    def test_floor_is_found(self):
        pts = move_sensor(scene(np.random.default_rng(0)), 0, 4, [0, 0, -1.8])
        R, h = estimate_floor(pts)
        levelled = pts @ R.T
        self.assertAlmostEqual(float(np.median(levelled[:3000, 2])), h, places=2)
        self.assertLess(float(np.std(levelled[:3000, 2])), 0.01)

    def test_invariant_to_sensor_height_tilt_and_position(self):
        pts = scene(np.random.default_rng(0))
        norm = Normalizer("floor", scale=2.0)
        a = norm(move_sensor(pts, 0, 0, [0, 0, -1.4]))
        b = norm(move_sensor(pts, 0, 5, [2.5, -1.0, -2.0]))     # higher, tilted, elsewhere
        np.testing.assert_allclose(a[:, 2], b[:, 2], atol=0.01)  # height above the floor
        np.testing.assert_allclose(a[:, :2], b[:, :2], atol=0.05)

    def test_heights_invariant_to_yaw(self):
        pts = scene(np.random.default_rng(0))
        norm = Normalizer("floor", scale=2.0)
        a = norm(move_sensor(pts, 0, 0, [0, 0, -1.7]))
        b = norm(move_sensor(pts, 70, 2, [1.0, 1.0, -1.7]))
        np.testing.assert_allclose(a[:, 2], b[:, 2], atol=0.01)


class DataTest(unittest.TestCase):
    def test_normalizer_fit_and_apply(self):
        tmp = tempfile.mkdtemp()
        try:
            write_frames(tmp, 3, np.random.default_rng(0))
            paths = sorted(os.path.join(tmp, f) for f in os.listdir(tmp))
            norm = Normalizer.fit(paths, "global")
            pts = np.concatenate([np.load(p)[:, :3] for p in paths])
            out = norm(pts)
            np.testing.assert_allclose(out.mean(0), 0, atol=1e-4)
            self.assertAlmostEqual(float(np.sqrt((out ** 2).sum(1).mean())), 1.0, places=4)
        finally:
            shutil.rmtree(tmp)

    def test_augment_rotation_preserves_geometry(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(50, 3)).astype(np.float32)
        out = augment(pts, rng, rotate_z=(0, 360), scale=None, jitter_std=0)
        np.testing.assert_allclose(np.linalg.norm(out[:, :2], axis=1),
                                   np.linalg.norm(pts[:, :2], axis=1), rtol=1e-5)
        np.testing.assert_allclose(out[:, 2], pts[:, 2])

    def test_dataset_fixed_size(self):
        tmp = tempfile.mkdtemp()
        try:
            write_frames(tmp, 2, np.random.default_rng(0), n_points=40)
            paths = sorted(os.path.join(tmp, f) for f in os.listdir(tmp))
            ds = PointCloudDataset(paths, 64, Normalizer.identity())   # fewer points than requested
            pts, lab = ds[0]
            self.assertEqual(tuple(pts.shape), (64, 3))
            self.assertEqual(tuple(lab.shape), (64,))
            ds = PointCloudDataset(paths, 64, Normalizer("floor", scale=2.0), pad=False,
                                   augmentation={"crop_xy": [1, 3], "crop_prob": 1.0})
            pts, lab = ds[0]
            self.assertEqual(len(np.unique(pts.numpy(), axis=0)), len(pts))  # no repeats
        finally:
            shutil.rmtree(tmp)


class InferenceTest(unittest.TestCase):
    def test_predict_every_point(self):
        model = DualStreamNetwork(num_classes=8, k=10).eval()
        pts = np.random.default_rng(0).normal(size=(150, 3)).astype(np.float32)
        pred = predict_frame(model, pts, 64, Normalizer.identity(), "cpu", exclude_class=7)
        self.assertEqual(pred.shape, (150,))
        self.assertTrue((pred >= 0).all() and (pred < 7).all())

    def test_checkpoint_roundtrip(self):
        tmp = tempfile.mkdtemp()
        try:
            model = DualStreamNetwork(num_classes=8, k=10).eval()
            norm = Normalizer("global", (1, 2, 3), 4.0)
            cfg = {"model": {"name": "dualstream", "args": {"num_classes": 8, "k": 10}}}
            path = os.path.join(tmp, "m.pth")
            save_checkpoint(path, model, norm, cfg, epoch=1)
            loaded, norm2, _ = load_checkpoint(path)
            np.testing.assert_allclose(norm2.center, [1, 2, 3])
            x = torch.randn(1, 3, 32)
            with torch.no_grad():
                torch.testing.assert_allclose(model(x), loaded.eval()(x))
        finally:
            shutil.rmtree(tmp)

    def test_legacy_checkpoint(self):
        """Checkpoints from the original code: DataParallel keys, no normalization."""
        tmp = tempfile.mkdtemp()
        try:
            model = DualStreamNetwork(num_classes=8)
            state = {"module." + k: v for k, v in model.state_dict().items()}
            path = os.path.join(tmp, "legacy.pth")
            torch.save({"epoch": 99, "model_state_dict": state}, path)
            _, norm, _ = load_checkpoint(path)
            self.assertEqual(norm.scale, 1.0)
        finally:
            shutil.rmtree(tmp)


class BaselinesTest(unittest.TestCase):
    def test_all_models_forward_and_reset_head(self):
        for name in ("dualstream", "pointnet", "dgcnn", "cnn_residual"):
            model = build_model(name, num_classes=8, k=10).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.randn(2, 3, 64)).shape), (2, 8, 64), name)
            model.reset_head(5)
            self.assertEqual(output_layer(model).out_channels, 5, name)
            for head in model.HEAD:
                self.assertTrue(hasattr(model, head), (name, head))


class DensityTest(unittest.TestCase):
    def test_voxel_majority_label(self):
        pts = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [1.0, 1.0, 1.0]], np.float32)
        lab = np.array([2, 2, 5, 3])
        out, out_lab = voxel_downsample(pts, 0.1, lab)
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(out_lab.tolist()), [2, 3])

    def test_nearest_chunk_is_contiguous(self):
        pts = np.random.default_rng(0).uniform(-5, 5, (2000, 3)).astype(np.float32)
        chunk = nearest_chunk(pts, 0, 100)
        self.assertEqual(len(set(chunk.tolist())), 100)
        r = np.linalg.norm(pts[chunk, :2] - pts[0, :2], axis=1).max()
        self.assertLess(r, 1.5)                         # ~ sqrt(100 / 2000 * 100 / pi)

    def test_pcd_roundtrip(self):
        tmp = tempfile.mkdtemp()
        try:
            pts = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
            lab = np.arange(50) % 8
            write_pcd(os.path.join(tmp, "a.pcd"), pts, lab)
            p2, l2 = read_pcd(os.path.join(tmp, "a.pcd"))
            np.testing.assert_allclose(p2, pts, atol=1e-5)
            np.testing.assert_array_equal(l2, lab)
        finally:
            shutil.rmtree(tmp)


class ConfigTest(unittest.TestCase):
    def test_override_numbers(self):
        from sim2real.config import parse_value
        self.assertEqual(parse_value("1e-4"), 1e-4)       # plain YAML would give a string
        self.assertEqual(parse_value("3"), 3)
        self.assertIs(parse_value("true"), True)
        self.assertIsNone(parse_value("null"))
        self.assertEqual(parse_value("outputs/x"), "outputs/x")


def labelled_scene(rng, n=400):
    """Floor (class 3) at z = 0 and a 1.6 m tall box (class 0), in a tilted sensor frame."""
    n_floor = n * 3 // 4
    floor = np.c_[rng.uniform(-3, 3, (n_floor, 2)), np.zeros(n_floor)]
    box = rng.uniform([0.5, 0.5, 0.0], [1.3, 1.3, 1.6], (n - n_floor, 3))
    pts = move_sensor(np.vstack([floor, box]), 30, 3, [1.0, -2.0, -1.7]).astype(np.float32)
    lab = np.r_[np.full(n_floor, 3), np.zeros(n - n_floor, dtype=np.int64)]
    return pts, lab


class TrainInferenceConsistencyTest(unittest.TestCase):
    def test_training_piece_equals_inference_input(self):
        """Without augmentation, a training sample and the inference input of the
        same frame must be the same point set (normalization applied identically)."""
        tmp = tempfile.mkdtemp()
        try:
            pts, lab = labelled_scene(np.random.default_rng(0))
            path = os.path.join(tmp, "f.npy")
            np.save(path, np.column_stack([pts, lab]).astype(np.float32))
            norm = Normalizer("floor", scale=2.5, voxel=0.05)
            ds = PointCloudDataset([path], 10000, norm, augmentation=None, pad=False)
            sample = ds[0][0].numpy()
            p2, _ = norm.downsample(pts, lab)
            inference = norm(p2)
            a = sample[np.lexsort(sample.T)]
            b = inference[np.lexsort(inference.T)]
            np.testing.assert_allclose(a, b, atol=1e-5)
        finally:
            shutil.rmtree(tmp)

    def test_chunked_prediction_uses_centred_pieces(self):
        model = DualStreamNetwork(num_classes=8, k=10).eval()
        pts = np.random.default_rng(0).uniform(-4, 4, (500, 3)).astype(np.float32)
        norm = Normalizer("floor", scale=2.0)
        p1 = predict_frame(model, pts, 200, norm, "cpu", seed=1)
        p2 = predict_frame(model, pts, 200, norm, "cpu", seed=1)
        np.testing.assert_array_equal(p1, p2)              # deterministic for a seed
        self.assertEqual(p1.shape, (500,))


class FineTuneTest(unittest.TestCase):
    def test_freezing(self):
        tmp = tempfile.mkdtemp()
        try:
            model = DualStreamNetwork(num_classes=8, k=10)
            cfg = {"model": {"name": "dualstream", "args": {"num_classes": 8, "k": 10}}}
            path = os.path.join(tmp, "m.pth")
            save_checkpoint(path, model, Normalizer("floor", scale=2.0, voxel=0.05), cfg, 1)

            ft_cfg = dict(cfg, finetune={"init_from": path, "freeze_encoders": True, "freeze_bn": True})
            model, _ = train_script.setup_model(ft_cfg, [], "cpu")
            trainable = {n for n, p in model.named_parameters() if p.requires_grad}
            self.assertTrue(all(n.startswith(("conv8", "conv9")) for n in trainable), trainable)
            self.assertTrue(any(n.startswith("conv9") for n in trainable))

            bn = model.dgcnn.bn1
            before = bn.running_mean.clone()
            model.train()
            train_script.freeze_batchnorm(model)
            x = torch.randn(2, 3, 64) * 5 + 3
            loss = model(x).sum()
            loss.backward()
            torch.testing.assert_allclose(bn.running_mean, before)   # statistics untouched
            self.assertIsNone(model.dgcnn.conv1[0].weight.grad)       # frozen: no gradient
            self.assertIsNotNone(model.conv9.weight.grad)
        finally:
            shutil.rmtree(tmp)


class LearningTest(unittest.TestCase):
    def test_model_can_fit_a_frame(self):
        """Gradient flows through both streams: a few steps fit floor vs box."""
        torch.manual_seed(0)
        pts, lab = labelled_scene(np.random.default_rng(0), n=300)
        norm = Normalizer("floor", scale=2.0)
        x = torch.from_numpy(norm(pts)).t().unsqueeze(0)
        y = torch.from_numpy(lab).unsqueeze(0)
        model = DualStreamNetwork(num_classes=8, k=10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(40):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x).argmax(1) == y).float().mean().item()
        self.assertGreater(acc, 0.9)


class CenterTest(unittest.TestCase):
    def test_center_and_none(self):
        pts = np.random.default_rng(0).uniform(-3, 3, (100, 3)).astype(np.float32) + [5, -7, 1]
        c = Normalizer("floor").recenter(pts)
        np.testing.assert_allclose(np.median(c[:, :2], axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(c[:, 2], pts[:, 2])
        np.testing.assert_allclose(Normalizer("none").recenter(pts), pts)


class MetricsTest(unittest.TestCase):
    def test_iou(self):
        cm = ConfusionMatrix(3, ignore_index=2)
        cm.update([0, 0, 1, 1, 0], [0, 1, 1, 1, 2])
        iou = cm.iou()
        self.assertAlmostEqual(iou[0], 1 / 2)
        self.assertAlmostEqual(iou[1], 2 / 3)
        self.assertTrue(np.isnan(iou[2]))
        self.assertAlmostEqual(cm.summary()["OA"], 3 / 4)


class EndToEndTest(unittest.TestCase):
    def test_train_and_evaluate(self):
        tmp = tempfile.mkdtemp()
        try:
            rng = np.random.default_rng(0)
            write_frames(os.path.join(tmp, "data", "sim", "s1"), 4, rng)
            write_frames(os.path.join(tmp, "data", "real", "r1"), 4, rng)
            py = sys.executable
            subprocess.check_call([py, os.path.join(ROOT, "tools", "make_splits.py"),
                                   "--root", "data", "--val", "0.25", "--test", "0.25",
                                   "--out", "splits"], cwd=tmp)
            subprocess.check_call([py, os.path.join(ROOT, "train.py"),
                                   "--config", os.path.join(ROOT, "configs", "sim_pretrain.yaml"),
                                   "--set", "data.train_split=splits/train.txt",
                                   "data.val_split=splits/val.txt",
                                   "data.test_split=splits/test.txt",
                                   "train.epochs=2", "train.batch_size=2",
                                   "train.num_workers=0", "data.num_points=128",
                                   "model.args.k=10", "output_dir=out"], cwd=tmp)
            subprocess.check_call([py, os.path.join(ROOT, "evaluate.py"),
                                   "--ckpt", "out/best.pth", "--split", "splits/test.txt",
                                   "--num-points", "128", "--out", "out/metrics.json"], cwd=tmp)
            self.assertTrue(os.path.exists(os.path.join(tmp, "out", "metrics.json")))

            subprocess.check_call([py, os.path.join(ROOT, "train.py"),
                                   "--config", os.path.join(ROOT, "configs", "finetune.yaml"),
                                   "--set", "finetune.init_from=out/best.pth",
                                   "data.train_split=splits/train.txt",
                                   "data.val_split=splits/val.txt",
                                   "data.test_split=splits/test.txt",
                                   "train.epochs=1", "train.num_workers=0",
                                   "data.num_points=128", "model.args.k=10",
                                   "output_dir=ft"], cwd=tmp)
            subprocess.check_call([py, os.path.join(ROOT, "train.py"),
                                   "--config", "out/config.yaml", "--resume", "out/last.pth",
                                   "--lr", "1e-4", "--set", "train.epochs=3"], cwd=tmp)
            _, norm_ft, _ = load_checkpoint(os.path.join(tmp, "ft", "best.pth"))
            _, norm_pre, _ = load_checkpoint(os.path.join(tmp, "out", "best.pth"))
            self.assertEqual(norm_ft.to_dict(), norm_pre.to_dict())   # never re-fitted
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
