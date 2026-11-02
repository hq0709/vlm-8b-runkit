"""Coarse-to-fine pointing data: zoomed crops around each training lesion, so the model learns the re-pointing step
that the inference harness (eval_zoom.py) asks of it. No new labels: crops and targets come from the expert boxes.

For each training slice with a lesion box: J crops of 448 / Z px (Z = 2) centred at the lesion centre plus a uniform
jitter of up to a quarter window (clamped to the image), resized to 448 px. The question is the crop prompt of
eval_zoom.py; the target is the box centre in crop pixels. The multi-task SFT sets are written as well:
  mt_train_nocrop.jsonl   laterality + quadrant + lobe + liver training splits (full slices only)
  mt_train_crop.jsonl     the same plus the crops (one shared crop set: quadrant uses the laterality slices)

    python build_crops.py
"""
import json, random
from pathlib import Path

from PIL import Image

D = Path(__file__).resolve().parent / "data/med"; OUT = D / "crops"; OUT.mkdir(exist_ok=True)
Z, J = 2.0, 2
CQ = ("This image is a {z:g}x zoomed-in crop of a medical image slice, centred near a lesion. Point to the centre of the lesion "
      "(the tumour or nodule) in this crop. Answer in exactly this format: <answer>lesion</answer><points>[[x,y]]</points> where "
      "(x, y) is in pixels of this 448 x 448 image, (0, 0) the top-left corner.")


def crops(x, rng):
    b = [v * 0.448 for v in x["boxes"][0]]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; w = 448 / Z; out = []
    im = Image.open(x["image"]).convert("RGB").resize((448, 448))
    for j in range(J):
        px, py = cx + rng.uniform(-w / 4, w / 4), cy + rng.uniform(-w / 4, w / 4)
        x0 = min(max(px - w / 2, 0), 448 - w); y0 = min(max(py - w / 2, 0), 448 - w); k = 448 / w
        p = OUT / f"{x['id']}_c{j}.png"
        im.crop((round(x0), round(y0), round(x0 + w), round(y0 + w))).resize((448, 448), Image.BICUBIC).save(p)
        cb = [max(0, (b[0] - x0) * k), max(0, (b[1] - y0) * k), min(448, (b[2] - x0) * k), min(448, (b[3] - y0) * k)]
        out.append({"id": f"{x['id']}_c{j}", "task": "crop", "split": "train", "patient": x.get("patient", x["id"]), "image": str(p),
                    "question": CQ.format(z=Z), "answer": "lesion", "scale": 1000 / 448, "boxes": [[round(v / 0.448) for v in cb]]})
    return out


if __name__ == "__main__":
    rng = random.Random(0)
    load = lambda f: [json.loads(l) for l in open(D / f) if l.strip()]
    vis, quad, lobe, liver = load("px_train_rl_vis.jsonl"), load("px_train_rl_quad.jsonl"), load("lobe_train.jsonl"), load("liver_train.jsonl")
    full = vis + quad + lobe + liver
    cr = [c for x in vis + lobe + liver for c in crops(x, rng)]
    for name, rows in (("mt_train_nocrop", full), ("mt_train_crop", full + cr)):
        rows = rows[:]; rng.shuffle(rows)
        with open(D / f"{name}.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        print(name, len(rows))
