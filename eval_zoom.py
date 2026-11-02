"""Point, zoom, re-point (training-free; after DeepScan's refocusing, arXiv 2603.03857). Pass 1 asks the question on the
full 448-px slice as usual. Pass 2 crops a window around the first point (448 / zoom px wide, clamped to the image),
resizes it to 448 px and asks for the lesion's centre in the crop (a crop-specific prompt), mapped back to the full
slice. The refined point is accepted only within --accept x crop width of the crop centre (a local correction);
otherwise, or if pass 2 gives no point, the pass-1 point is kept. The stated answer stays the one from pass 1; the
derived answer is read from the final point by the usual program.

    python eval_zoom.py --model M [--adapter A] --data data/med/lobe_test.jsonl --zoom 2 --out results/X_zoom2.jsonl
"""
import argparse, json
from pathlib import Path

import torch
from PIL import Image

from eval_ground import load, parse, score


def generate(model, proc, ims, questions, max_new=160):
    texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
                                      add_generation_prompt=True, tokenize=False) for q in questions]
    enc = proc(text=texts, images=[[im] for im in ims], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def crop_box(px, py, zoom, size=448):
    w = size / zoom; x0 = min(max(px - w / 2, 0), size - w); y0 = min(max(py - w / 2, 0), size - w)
    return x0, y0, w


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accept", type=float, default=0.3)
    ap.add_argument("--adaptive", action="store_true", help="zoom only where pass 1 ends with <zoom>yes</zoom>")
    a = ap.parse_args()
    CQ = ("This image is a {z:g}x zoomed-in crop of a medical image slice, centred near a lesion. Point to the centre of the lesion "
          "(the tumour or nodule) in this crop. Answer in exactly this format: <answer>lesion</answer><points>[[x,y]]</points> where "
          "(x, y) is in pixels of this 448 x 448 image, (0, 0) the top-left corner.")
    items = [json.loads(l) for l in open(a.data) if l.strip()]
    model, proc = load(a.model, a.adapter)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            full = [Image.open(x["image"]).convert("RGB").resize((448, 448)) for x in chunk]
            t1 = generate(model, proc, full, [x["question"] for x in chunk])
            crops, meta = [], []
            for x, im, t in zip(chunk, full, t1):
                _, pts, _ = parse(t, 1.0)                      # pass-1 points in the model's own 448-px pixels
                if pts and (not a.adaptive or "<zoom>yes</zoom>" in t):
                    x0, y0, w = crop_box(pts[0][0], pts[0][1], a.zoom)
                    crops.append(im.crop((round(x0), round(y0), round(x0 + w), round(y0 + w))).resize((448, 448), Image.BICUBIC))
                    meta.append((x0, y0, w))
                else:
                    crops.append(None); meta.append(None)
            idx = [j for j, c in enumerate(crops) if c is not None]
            t2 = dict(zip(idx, generate(model, proc, [crops[j] for j in idx], [CQ.format(z=a.zoom)] * len(idx)))) if idx else {}
            for j, (x, t) in enumerate(zip(chunk, t1)):
                final = t
                if j in t2:
                    _, p2, _ = parse(t2[j], 1.0)
                    if p2 and ((p2[0][0] - 224) ** 2 + (p2[0][1] - 224) ** 2) ** 0.5 <= a.accept * 448:
                        x0, y0, w = meta[j]; k = w / 448
                        ans = parse(t, 1.0)[0] or ""
                        final = f"<answer>{ans}</answer><points>[[{round(x0 + p2[0][0] * k)}, {round(y0 + p2[0][1] * k)}]]</points>"
                f.write(json.dumps({"id": x["id"], "task": x["task"], "organ": x.get("organ", ""), "split": x["split"], "text": final,
                                    "text1": t, "text2": t2.get(j), **score(x, final, x.get("scale", 1.0)),
                                    **{f"{k}@1": v for k, v in score(x, t, x.get("scale", 1.0)).items()}}) + "\n")
    rows = [json.loads(l) for l in open(a.out)]
    if a.adaptive:
        print(f"zoom requested: {sum('<zoom>yes</zoom>' in r['text1'] for r in rows)}/{len(rows)}, "
              f"refined point accepted: {sum(r['text'] != r['text1'] for r in rows)}")
    for key in sorted({(r["split"], r["task"]) for r in rows}):
        rs = [r for r in rows if (r["split"], r["task"]) == key]
        for suf, name in (("@1", "pass 1"), ("", f"zoom {a.zoom:g}")):
            print(f"{key[0]:5s} {key[1]:11s}{name:>8s} n={len(rs):4d} | " + "  ".join(f"{k} {100 * sum(r[k + suf] for r in rs) / len(rs):5.1f}%" for k in ("format", "answer", "grounded", "both", "lucky", "derived", "dboth")))
