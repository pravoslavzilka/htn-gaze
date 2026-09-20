#!/usr/bin/env python3
"""Label real object crops with OMNI (offline, never in the live loop), for training the shape net.

Crops come from `tools/collect.py --label auto` (the real mixed table, hands and all). They're sent in
numbered 4x4 contact sheets; each crop is labelled in --passes different sheets (shuffled), and only
crops every pass agrees on are kept. Disagreements and "unsure" go to data/review/ for a human look.

  python tools/omni_label.py                        # data/unlabeled -> data/real/<label>/
  python tools/omni_label.py --limit 64 --dry-run   # try a few sheets, print, move nothing
Then: .venv-train/bin/python tools/train_shape.py --real data/real --arch mobilenet
Needs OMNI_API_KEY (env or .env). Cost: passes x crops / 16 calls.
"""
import argparse
import base64
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from outer_vision import config, omni, shape_net  # noqa: E402

LABELS = ["round", "square", "cylinder", "triangle", "reject"]
TILE, GRID = 160, 4

PROMPT = """You label training images for a shape classifier. The picture is a 4x4 sheet of numbered tiles.
Each tile is a crop from a head-mounted camera looking down at a table, centred on one candidate object.
For EVERY tile number, pick one label for the object in the CENTRE of the tile:
- round: a ball/sphere, or a flat round disc or puck
- square: a cube or box-like block (flat faces, corners), from any angle
- cylinder: a can, tube, cup or roll (straight parallel sides with round ends), standing or lying down
- triangle: a wedge, a folded card, a triangular block or prism: the outline is a triangle, three corners
- reject: anything else: a hand, fingers, arm, pen, paper, cable, shadow, several objects, an object cut off
  so its shape can't be told, or nothing clear
- unsure: you can't tell between two of the above
Colour does not matter. Reply with ONLY a JSON object mapping every tile number to a label, e.g.
{"1": "round", "2": "reject", ...}"""


def sheet(paths):
    img = np.full((GRID * TILE, GRID * TILE, 3), 40, np.uint8)
    for i, p in enumerate(paths):
        r, c = divmod(i, GRID)
        tile = cv2.resize(cv2.imread(str(p)), (TILE - 6, TILE - 6), interpolation=cv2.INTER_CUBIC)
        img[r * TILE + 3:(r + 1) * TILE - 3, c * TILE + 3:(c + 1) * TILE - 3] = tile
        cv2.rectangle(img, (c * TILE + 3, r * TILE + 3), (c * TILE + 31, r * TILE + 25), (255, 255, 255), -1)
        cv2.putText(img, str(i + 1), (c * TILE + 6, r * TILE + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def ask(client, paths):
    jpg = cv2.imencode(".jpg", sheet(paths), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    msgs = [{"role": "system", "content": PROMPT}, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}},
        {"type": "text", "text": f"Tiles 1 to {len(paths)}."}]}]
    try:
        r = omni.parse_reply("".join(d for k, d in client.stream(msgs) if k == "text"))
    except omni.OmniError as e:
        print(f"  sheet failed: {e}", flush=True)
        return {}
    return {p: str(r.get(str(i + 1), "unsure")).lower() for i, p in enumerate(paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/unlabeled"))
    ap.add_argument("--out", default=str(ROOT / "data/real"))
    ap.add_argument("--review", default=str(ROOT / "data/review"))
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    config.load_env(ROOT / ".env")
    cfg = config.load(str(ROOT / "config.json"))
    o = cfg["omni"]
    if not os.environ.get("OMNI_API_KEY"):
        sys.exit("set OMNI_API_KEY (env or .env) first")
    client = omni.OmniClient(os.environ.get("OMNI_BASE_URL", o["base_url"]), os.environ["OMNI_API_KEY"],
                             os.environ.get("OMNI_MODEL", o["model"]), o["voice"], timeout=60)
    paths = sorted(Path(args.src).glob("*.png"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        sys.exit(f"no crops in {args.src}; run tools/collect.py --label auto first")
    print(f"{len(paths)} crops, {args.passes} passes, ~{args.passes * -(-len(paths) // 16)} OMNI calls", flush=True)

    votes = {p: [] for p in paths}
    rng = random.Random(0)
    for n in range(args.passes):
        order = paths[:]
        rng.shuffle(order)                     # a crop lands in a different tile/sheet each pass
        batches = [order[i:i + GRID * GRID] for i in range(0, len(order), GRID * GRID)]
        with ThreadPoolExecutor(args.workers) as ex:
            for res in ex.map(lambda b: ask(client, b), batches):
                for p, lab in res.items():
                    votes[p].append(lab)
        print(f"pass {n + 1}/{args.passes} done", flush=True)

    net = shape_net.load(cfg)                  # current model's opinion, for a disagreement report
    counts, kept, review = {l: 0 for l in LABELS}, 0, 0
    agree_net, manifest = 0, []
    for p, v in votes.items():
        lab = v[0] if len(v) == args.passes and len(set(v)) == 1 and v[0] in LABELS else None
        pred = None
        if net is not None:
            pred = net.labels[int(net.probs([cv2.imread(str(p))]).argmax())]
        manifest.append({"file": p.name, "votes": v, "label": lab, "current_model": pred})
        if lab is None:
            review += 1
            dest = Path(args.review)
        else:
            kept += 1
            counts[lab] += 1
            agree_net += pred == lab
            dest = Path(args.out) / lab
        if not args.dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), dest / f"omni_{p.name}")
    if not args.dry_run:
        with open(Path(args.out).parent / "omni_labels.jsonl", "a") as f:
            for m in manifest:
                f.write(json.dumps(m) + "\n")
    else:
        for m in manifest[:32]:
            print(" ", m)
    print(f"kept {kept} {counts}, to review {review}")
    if net is not None and kept:
        print(f"current shape model agrees with OMNI on {agree_net / kept:.0%} of kept crops "
              "(low = the synthetic-only model is struggling on real objects; retrain)")


if __name__ == "__main__":
    main()
