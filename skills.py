"""Reading skills executed by the harness (shared by skill-use RL in train_grpo.py and by eval_skills.py).

Pass 1 gives an answer, a point and the decisions <zoom>yes|no</zoom><window>yes|no</window>. Then, in order:
  window  the same slice in the alternative CT window (alt_window.json) is shown with the window prompt rewards.WP; the
          new point replaces the old one if it lies within 0.25 x 448 px of it (the same lesion, seen better). MRI has
          no alternative window: the call is a no-op.
  zoom    a 2x crop around the current point of the current image (the alternative window if it was used) is shown
          with the crop prompt; the new point, mapped back, replaces the old one if it lies within 0.35 x 448 px of the
          crop centre.
The final text keeps the pass-1 answer and carries the final point; the derived answer is read from it as usual.
"""
import json, os

import torch
from PIL import Image

from build_crops import CQ
from eval_ground import parse
from eval_zoom import crop_box
from rewards import WP

ALT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/med/alt_window.json")))


class SkillRunner:
    def __init__(self, model, proc, zoom=2.0, accept_zoom=0.35, accept_window=0.25):
        self.model, self.proc, self.zoom = model, proc, zoom
        self.accept_zoom, self.accept_window = accept_zoom, accept_window

    def _ask(self, ims, questions):
        texts = [self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
                                               add_generation_prompt=True, tokenize=False) for q in questions]
        enc = self.proc(text=texts, images=[[im] for im in ims], return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=40, do_sample=False)
        return self.proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def run(self, items, texts, zoom_flags, window_flags):
        """Final texts after executing the requested skills; also returns, per item, which skills changed the point."""
        state = []
        for it, t in zip(items, texts):
            _, pts, _ = parse(t, 1.0)
            img = Image.open(it["image"]).convert("RGB").resize((448, 448))
            state.append({"pt": list(pts[0]) if pts else None, "img": img, "ans": parse(t, 1.0)[0] or "", "used": []})
        # window
        jobs = [i for i, s in enumerate(state) if window_flags[i] and s["pt"] is not None and items[i]["image"] in ALT]
        if jobs:
            ims = [Image.open(ALT[items[i]["image"]]).convert("RGB").resize((448, 448)) for i in jobs]
            qs = [WP.format(w="abdominal" if items[i]["task"] == "segment" else "soft-tissue") for i in jobs]
            for i, im, t2 in zip(jobs, ims, self._ask(ims, qs)):
                state[i]["img"] = im
                _, p2, _ = parse(t2, 1.0)
                if p2 and ((p2[0][0] - state[i]["pt"][0]) ** 2 + (p2[0][1] - state[i]["pt"][1]) ** 2) ** 0.5 <= self.accept_window * 448:
                    state[i]["pt"] = list(p2[0]); state[i]["used"].append("window")
        # zoom
        jobs = [i for i, s in enumerate(state) if zoom_flags[i] and s["pt"] is not None]
        if jobs:
            boxes = [crop_box(state[i]["pt"][0], state[i]["pt"][1], self.zoom) for i in jobs]
            ims = [state[i]["img"].crop((round(x0), round(y0), round(x0 + w), round(y0 + w))).resize((448, 448), Image.BICUBIC)
                   for i, (x0, y0, w) in zip(jobs, boxes)]
            for i, (x0, y0, w), t2 in zip(jobs, boxes, self._ask(ims, [CQ.format(z=self.zoom)] * len(jobs))):
                _, p2, _ = parse(t2, 1.0)
                if p2 and ((p2[0][0] - 224) ** 2 + (p2[0][1] - 224) ** 2) ** 0.5 <= self.accept_zoom * 448:
                    k = w / 448; state[i]["pt"] = [x0 + p2[0][0] * k, y0 + p2[0][1] * k]; state[i]["used"].append("zoom")
        out = []
        for s, t in zip(state, texts):
            out.append(t if s["pt"] is None else f"<answer>{s['ans']}</answer><points>[[{round(s['pt'][0])}, {round(s['pt'][1])}]]</points>")
        return out, [s["used"] for s in state]
