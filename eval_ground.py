"""Answer-and-point evaluation on the medical suite (build_med.py): is the answer right, and does the model point to the
right place?

  answer     exact match (left/right, yes/no, count)
  grounded   laterality: the first point lies inside the nodule disc
             count: points match lesion boxes one-to-one with F1 = 1 (every lesion pointed at, no stray points)
             detect: GT yes -> at least one point and every point inside a lesion box; GT no -> no points
  both       answer and grounded            lucky = answer right but not grounded (right for the wrong reason)

    python eval_ground.py --model M --data data/med/test.jsonl --out results/M.jsonl [--limit N]
"""
import argparse, json, re
from pathlib import Path

import torch
from PIL import Image

ANS = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
PTS = re.compile(r"<points>\s*(\[.*?\])\s*</points>", re.S | re.I)
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse(text, scale=1.0):
    """Strict tags if present; otherwise the leading answer and any points in the text (attribute or bracket form)."""
    a = ANS.search(text); p = PTS.search(text); fmt = bool(a and p)
    if a:
        ans = a.group(1).strip().lower().rstrip(".")
    else:
        ans = re.split(r"[<\n]", text.strip(), 1)[0].strip().lower().rstrip(".") or None
    if p:
        nums = [float(x) for x in NUM.findall(p.group(1))]
        pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    else:
        attr = re.findall(r'x\d*\s*=\s*"?(-?\d+(?:\.\d+)?)"?\s*y\d*\s*=\s*"?(-?\d+(?:\.\d+)?)"?', text)
        pairs = re.findall(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]", text)
        pts = [(float(x), float(y)) for x, y in (attr or pairs)]
    return ans, [(x * scale, y * scale) for x, y in pts], fmt


def inside(p, b):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def score(item, text, scale=1.0):
    ans, pts, fmt = parse(text, scale)
    gt = item["answer"]; t = item["task"]
    if t == "count":
        m = re.search(r"\d+", ans or ""); a_ok = bool(m) and m.group(0) == gt
        used, hit = set(), 0
        for p in pts:
            j = next((j for j, b in enumerate(item["boxes"]) if j not in used and inside(p, b)), None)
            if j is not None:
                used.add(j); hit += 1
        g_ok = hit == len(item["boxes"]) == len(pts)
    elif t == "laterality_vis":
        a_ok = ans is not None and gt in ans and ({"left": "right", "right": "left"}[gt] not in ans)
        g_ok = bool(pts) and inside(pts[0], item["boxes"][0])
    elif t == "lobe":
        from rewards import lobe_ok
        a_ok = ans is not None and lobe_ok(gt, ans)
        g_ok = bool(pts) and inside(pts[0], item["boxes"][0])
    elif t == "segment":
        from rewards import seg_ok
        a_ok = ans is not None and seg_ok(gt, ans)
        g_ok = bool(pts) and inside(pts[0], item["boxes"][0])
    elif t == "quadrant":
        from rewards import quad_ok
        a_ok = ans is not None and quad_ok(gt, ans)
        g_ok = bool(pts) and inside(pts[0], item["boxes"][0])
    elif t == "laterality":
        a_ok = ans is not None and gt in ans and ({"left": "right", "right": "left"}[gt] not in ans)
        cx, cy, r = item["discs"][0]
        g_ok = bool(pts) and ((pts[0][0] - cx) ** 2 + (pts[0][1] - cy) ** 2) ** 0.5 <= r
    else:
        a_ok = ans is not None and ans.startswith(gt)
        g_ok = (len(pts) == 0) if gt == "no" else (len(pts) > 0 and all(any(inside(p, b) for b in item["boxes"]) for p in pts))
    from rewards import derived_ok
    d_ok = bool(derived_ok(item, pts))
    return {"format": fmt, "answer": a_ok, "grounded": g_ok, "both": a_ok and g_ok, "lucky": a_ok and not g_ok,
            "derived": d_ok, "dboth": d_ok and g_ok}


def load(name, adapter=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(name, min_pixels=448 * 448, max_pixels=448 * 448)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa").eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    return model, proc


def generate(model, proc, items, max_new=160):
    texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": x["question"]}]}],
                                      add_generation_prompt=True, tokenize=False) for x in items]
    ims = [[Image.open(x["image"]).convert("RGB").resize((448, 448))] for x in items]
    enc = proc(text=texts, images=ims, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/med/test.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max_new", type=int, default=160, help="generation budget; reasoning formats need their training length")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.data) if l.strip()]
    if a.limit:
        items = items[:a.limit]
    model, proc = load(a.model, a.adapter)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            for x, t in zip(chunk, generate(model, proc, chunk, a.max_new)):
                f.write(json.dumps({"id": x["id"], "task": x["task"], "organ": x.get("organ", ""), "split": x["split"], "text": t,
                                    **score(x, t, x.get("scale", 1.0)), **{f"{k}@448": v for k, v in score(x, t, 1000 / 448).items()}}) + "\n")
    rows = [json.loads(l) for l in open(a.out)]
    for key in sorted({(r["split"], r["task"]) for r in rows}):
        rs = [r for r in rows if (r["split"], r["task"]) == key]
        for suf in ("", "@448"):
            print(f"{key[0]:5s} {key[1]:11s}{suf:5s} n={len(rs):4d} | " + "  ".join(f"{k} {100 * sum(r[k + suf] for r in rs) / len(rs):5.1f}%" for k in ("format", "answer", "grounded", "both", "lucky", "derived", "dboth")))
