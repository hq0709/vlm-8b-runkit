"""Evaluate a reading-skill model under a skill policy. Pass 1 on the full slice (question with both skill instructions,
*_sq.jsonl), then the skills (skills.py) executed by the same model:

  model    the model's own decisions (<zoom>, <window> tags)      never   no skill (pass-1 point)
  zoom     always zoom                                             window  always window (CT only)
  both     always window, then zoom

    python eval_skills.py --model M [--adapter A] --data data/med/lobe_test_sq.jsonl --policy model --out results/X.jsonl
"""
import argparse, json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

from eval_ground import load, score
from skills import SkillRunner


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
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", choices=["model", "never", "zoom", "window", "both"], default="model")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.data) if l.strip()]
    model, proc = load(a.model, a.adapter)
    runner = SkillRunner(model, proc)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    calls = Counter()
    with open(a.out, "w") as f:
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            t1 = [t.strip() for t in generate(model, proc, chunk)]
            if a.policy == "model":
                zf = ["<zoom>yes</zoom>" in t for t in t1]; wf = ["<window>yes</window>" in t for t in t1]
            else:
                zf = [a.policy in ("zoom", "both")] * len(chunk); wf = [a.policy in ("window", "both")] * len(chunk)
            final, used = runner.run(chunk, t1, zf, wf)
            for x, t, fin, z, w, u in zip(chunk, t1, final, zf, wf, used):
                calls["zoom"] += z; calls["window"] += w; calls["zoom used"] += "zoom" in u; calls["window used"] += "window" in u
                f.write(json.dumps({"id": x["id"], "task": x["task"], "organ": x.get("organ", ""), "split": x["split"], "text": fin,
                                    "text1": t, "zoom": z, "window": w, "used": u, **score(x, fin, x.get("scale", 1.0)),
                                    **{f"{k}@1": v for k, v in score(x, t, x.get("scale", 1.0)).items()}}) + "\n")
    rows = [json.loads(l) for l in open(a.out)]
    for key in sorted({(r["split"], r["task"]) for r in rows}):
        rs = [r for r in rows if (r["split"], r["task"]) == key]
        print(f"{key[0]:5s} {key[1]:11s} {a.policy:>6s} n={len(rs):4d} | " + "  ".join(f"{k} {100 * sum(r[k] for r in rs) / len(rs):5.1f}%" for k in ("format", "answer", "grounded", "both", "lucky", "derived", "dboth")))
    print("skill calls:", dict(calls), "of", len(rows))
