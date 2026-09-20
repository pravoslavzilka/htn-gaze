#!/usr/bin/env python3
"""Train the shape classifier and export it to ONNX (run by OpenCV DNN at runtime).

Runs on the Mac in the training env (torch is NOT needed at runtime):
  uv venv --python 3.12 .venv-train && uv pip install --python .venv-train/bin/python -r requirements-train.txt
  .venv-train/bin/python tools/train_shape.py                      # synthetic only (bootstrap)
  .venv-train/bin/python tools/train_shape.py --real data/real     # + crops from collect.py / omni_label.py

Real crops live in data/real/<round|square|cylinder|triangle|reject>/*.png and are weighted up.
--arch mobilenet (default): ImageNet-pretrained MobileNetV3-small at 96 px, 1.5M params. Pretrained
  features carry over from synthetic renders to real objects far better than a net trained from scratch.
  ~1.2 ms/crop in OpenCV on the Mac.
--arch tiny: the original 70k-param CNN at 64 px (~0.3 ms/crop) if the Pi can't keep up.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outer_vision import synthetic  # noqa: E402

LABELS = ["round", "square", "cylinder", "triangle", "reject"]
SIZE = 64          # set from --arch in main()
ARCH_SIZE = {"tiny": 64, "mobilenet": 96}


class ShapeCNN(nn.Module):
    """~70k params: 4 stride-2 blocks (64 -> 4 px), global average pool, linear head."""

    def __init__(self, n=len(LABELS)):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.features = nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 96))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(96, n)

    def forward(self, x):                     # x: N,3,64,64 RGB in 0..255
        x = self.features(x * (1.0 / 255.0))
        return self.head(torch.flatten(self.pool(x), 1))


class PretrainedNet(nn.Module):
    """MobileNetV3-small (ImageNet weights) with a new head, one logit per LABELS entry. Takes the same
    input as ShapeCNN (RGB 0..255) and normalises inside, so the runtime (shape_net.py) doesn't change."""

    def __init__(self, n=len(LABELS)):
        super().__init__()
        import torchvision
        self.m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.m.classifier[-1] = nn.Linear(self.m.classifier[-1].in_features, n)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return self.m((x - self.mean) / self.std)


def synth_set(n, seed):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(n):
        lab = i % len(LABELS)
        xs.append(synthetic.random_crop_sample(rng, LABELS[lab], SIZE))
        ys.append(lab)
    return np.stack(xs), np.array(ys)


def track_key(name: str) -> str:
    """collect.py names crops [omni_]<session>_<frame>_<track>.png: crops of one track are near-copies."""
    parts = name.removeprefix("omni_").rsplit(".", 1)[0].split("_")
    return f"{parts[0]}_{parts[-1]}" if len(parts) >= 3 else name


def real_set(root):
    xs, ys, groups = [], [], []
    for lab, name in enumerate(LABELS):
        for f in sorted(Path(root, name).glob("*.png")):
            im = cv2.imread(str(f))
            if im is not None:
                xs.append(cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
                ys.append(lab)
                groups.append(f"{name}/{track_key(f.name)}")
    if not xs:
        return np.zeros((0, SIZE, SIZE, 3), np.uint8), np.zeros(0, int), []
    return np.stack(xs), np.array(ys), groups


def augment(batch_bgr, rng):
    """Hue rotation (shape must not depend on colour), brightness, flips, and real-camera degradations
    (motion/focus blur, sensor noise, low resolution). Applied to every batch."""
    out = []
    for im in batch_bgr:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.integers(180)) % 180
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.75, 1.2), 0, 255)
        im = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if rng.random() < 0.5:
            im = im[:, ::-1]
        if rng.random() < 0.3:
            k = int(rng.choice([3, 5]))
            im = cv2.GaussianBlur(im, (k, k), 0)
        if rng.random() < 0.3:                       # small objects far away: few real pixels
            s = int(im.shape[0] * rng.uniform(0.35, 0.7))
            im = cv2.resize(cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA), im.shape[:2][::-1])
        if rng.random() < 0.3:
            im = np.clip(im + rng.normal(0, rng.uniform(3, 10), im.shape), 0, 255).astype(np.uint8)
        out.append(np.ascontiguousarray(im))
    return np.stack(out)


def to_tensor(bgr):
    return torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]).transpose(0, 3, 1, 2).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="folder with real crops per label")
    ap.add_argument("--synthetic", type=int, default=24000)
    ap.add_argument("--arch", choices=sorted(ARCH_SIZE), default="mobilenet")
    ap.add_argument("--epochs", type=int, default=None, help="default 14 (tiny) / 8 (mobilenet)")
    ap.add_argument("--out", default=str(ROOT / "models/shape"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global SIZE
    SIZE = ARCH_SIZE[args.arch]
    args.epochs = args.epochs or (14 if args.arch == "tiny" else 8)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    t0 = time.time()
    xs, ys = synth_set(args.synthetic, args.seed)
    vx, vy = synth_set(2000, 10_000 + args.seed)
    print(f"synthetic: {len(xs)} train / {len(vx)} val crops in {time.time() - t0:.0f}s")
    rvx = rvy = None
    if args.real:
        rx, ry, groups = real_set(args.real)
        print(f"real: {len(rx)} crops from {len(set(groups))} object tracks "
              + str({l: int((ry == i).sum()) for i, l in enumerate(LABELS)}))
        if len(rx):
            # Hold out whole tracks: crops of one object 0.25 s apart are near-copies, and a random split
            # would put copies of validation crops in training (inflated real-val).
            keys = sorted(set(groups))
            val_keys = set(rng.permutation(keys)[:max(1, len(keys) // 6)])
            is_val = np.array([g in val_keys for g in groups])
            rvx, rvy = rx[is_val], ry[is_val]
            rx, ry = rx[~is_val], ry[~is_val]
            reps = max(1, len(xs) // (2 * max(1, len(rx))))   # real ≈ 1/3 of each epoch
            xs, ys = np.concatenate([xs] + [rx] * reps), np.concatenate([ys] + [ry] * reps)

    model = (ShapeCNN() if args.arch == "tiny" else PretrainedNet()).to(dev)
    lr = 3e-3 if args.arch == "tiny" else 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bs = 256 if args.arch == "tiny" else 128
    steps = args.epochs * (len(xs) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    step = 0
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(xs))
        tot = 0.0
        for i in range(0, len(perm) - bs + 1, bs):
            b = perm[i:i + bs]
            x = to_tensor(augment(xs[b], rng)).to(dev)
            y = torch.from_numpy(ys[b]).to(dev)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot += loss.item()
        acc = evaluate(model, vx, vy, dev)
        msg = f"epoch {ep + 1:2d}/{args.epochs}  loss {tot / max(1, len(perm) // bs):.3f}  synth-val {acc:.3f}"
        if rvx is not None:
            msg += f"  real-val {evaluate(model, rvx, rvy, dev):.3f}"
        print(msg, flush=True)

    model = model.cpu().eval()
    export(model, Path(args.out), {
        "labels": LABELS, "input_size": SIZE, "arch": args.arch, "synthetic_val_acc": round(evaluate(model, vx, vy, "cpu"), 4),
        "real_val_acc": None if rvx is None else round(evaluate(model, rvx, rvy, "cpu"), 4),
        "trained": time.strftime("%Y-%m-%d %H:%M"), "real_data": args.real,
    }, vx[:64])


@torch.no_grad()
def evaluate(model, x, y, dev):
    was_training = model.training
    model.eval()
    pred = []
    for i in range(0, len(x), 512):
        pred.append(model(to_tensor(x[i:i + 512]).to(dev)).argmax(1).cpu().numpy())
    model.train(was_training)
    return float((np.concatenate(pred) == y).mean())


def export(model, out: Path, meta, check_bgr):
    out.mkdir(parents=True, exist_ok=True)
    model.eval()   # BatchNorm must be frozen for export
    ex = torch.rand(1, 3, SIZE, SIZE) * 255
    torch.onnx.export(model, ex, str(out / "shape.onnx"), input_names=["in0"], output_names=["out0"],
                      dynamic_axes={"in0": {0: "n"}, "out0": {0: "n"}}, opset_version=13, dynamo=False)
    (out / "labels.json").write_text(json.dumps(meta, indent=2) + "\n")

    # The exported model must agree with torch, or the runtime silently misbehaves.
    ref = model(to_tensor(check_bgr)).detach().numpy()
    cvnet = cv2.dnn.readNetFromONNX(str(out / "shape.onnx"))
    cvnet.setInput(cv2.dnn.blobFromImages(list(check_bgr), 1.0, (SIZE, SIZE), swapRB=True))
    got = cvnet.forward()
    d = float(np.abs(got - ref).max())
    agree = float((got.argmax(1) == ref.argmax(1)).mean())
    print(f"exported to {out}: onnx/opencv max|diff| {d:.1e}, argmax agree {agree:.0%}")
    assert agree == 1.0 and d < 1e-3, "exported model disagrees with torch"
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
