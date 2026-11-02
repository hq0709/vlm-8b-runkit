"""MedGround-R1 (MICCAI 2025) spatial-semantic reward, reproduced for answer-and-point outputs on our benchmark.

MedGround-R1 rewards a grounding completion by spatial accuracy when it overlaps the target, and by semantic consistency
between the predicted region and the referring expression when it does not (so that spatially negative completions still
receive graded feedback). With point outputs: spatial = 1 if the point lies in the lesion box; otherwise the reward is
0.5 x sem, where sem compares a frozen BiomedCLIP's image-text similarity of a crop around the point with that of the whole
image (sigmoid of the margin). The total reward adds answer accuracy and format, as in the original.
"""
import math

import torch
from PIL import Image

FINDING = {"brain": "a brain tumour", "lung": "a lung tumour", "liver": "a liver tumour"}


def finding_text(item):
    if item.get("organ") == "chest_xray":
        return "a " + item["finding"]
    if "luna" in item["id"]:
        return "a lung nodule"
    if "BRATS" in item["id"]:
        return FINDING["brain"]
    if item["id"].startswith("liv_"):
        return FINDING["liver"]
    return FINDING["lung"]


class SemanticScorer:
    def __init__(self, device):
        import open_clip
        self.model, self.pre = open_clip.create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.tok = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.model = self.model.to(device).eval(); self.device = device; self.cache = {}

    @torch.no_grad()
    def _img(self, ims):
        x = torch.stack([self.pre(im) for im in ims]).to(self.device)
        f = self.model.encode_image(x); return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _txt(self, texts):
        f = self.model.encode_text(self.tok(texts).to(self.device)); return f / f.norm(dim=-1, keepdim=True)

    def scores(self, items, pts_list, half=48):
        """sem in (0, 1) per item (0.5 when no point): sigmoid(20 x (cos(crop, text) - cos(image, text)))."""
        out = [0.0] * len(items); jobs = [(i, it, p[0]) for i, (it, p) in enumerate(zip(items, pts_list)) if p]
        if not jobs:
            return out
        crops, fulls, texts = [], [], []
        for i, it, (x, y) in jobs:
            im = Image.open(it["image"]).convert("RGB").resize((448, 448))
            px, py = x * 0.448, y * 0.448                                    # points are on [0, 1000]
            box = (max(0, round(px - half)), max(0, round(py - half)), min(448, round(px + half)), min(448, round(py + half)))
            crops.append(im.crop(box) if box[2] > box[0] and box[3] > box[1] else im); fulls.append(im); texts.append(finding_text(it))
        fc, ff, ft = self._img(crops), self._img(fulls), self._txt(texts)
        margin = ((fc * ft).sum(-1) - (ff * ft).sum(-1)).tolist()
        for (i, _, _), m in zip(jobs, margin):
            out[i] = 1 / (1 + math.exp(-20 * m))
        return out
