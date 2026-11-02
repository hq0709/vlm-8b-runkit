"""Grounding-aware RL (GRPO) on the medical answer-and-point suite.

LoRA on the language model only (the vision tower is frozen), generation with vLLM. Reward modes:
  grounded  A * B + 0.1 * format    (answer right AND pointing right)
  answer    A + 0.1 * format        (control: answer only)
The answer and grounding components are logged separately (weight 0) so both curves are visible in either mode.

    accelerate launch --num_processes 3 train_grpo.py --model Qwen/Qwen2.5-VL-3B-Instruct --mode grounded --out runs/qwen3b_grounded
"""
import argparse, json, os, sys

from datasets import Dataset, Image as HFImage
from peft import LoraConfig
from transformers import AutoProcessor
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ground import inside, parse
from rewards import LOBE_NAMES, SEG_NAMES, STRICT, STRICT_SKILLS, STRICT_ZOOM, answer_ok, consistency, derived_answer, derived_ok, grounding, reward

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
ap.add_argument("--mode", choices=["grounded", "answer", "consistent", "selfcons", "derived", "derived_cf", "derived_box",
                                   "derived_vote", "stated_vote", "derived_v2", "derived_v2_nobox", "derived_v2_semi", "derived_v2_zoom", "derived_v2_skills",
                                   "derived_v3", "derived_v3_nobox", "derived_v3_semi", "derived_v3_skills",
                                   "additive", "deepeyes", "medground", "vigorl", "grit"], default="grounded")
ap.add_argument("--verifier", default="Qwen/Qwen2.5-VL-3B-Instruct")
ap.add_argument("--nec", choices=["soft", "vote"], default="vote")
ap.add_argument("--scale_rewards", default="group")
ap.add_argument("--beta", type=float, default=0.0)
ap.add_argument("--out", required=True)
ap.add_argument("--data", default="data/med/train448.jsonl")
ap.add_argument("--max_steps", type=int, default=800)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--num_gen", type=int, default=8)
ap.add_argument("--per_device", type=int, default=8)
ap.add_argument("--grad_accum", type=int, default=1,
                help="gradient accumulation steps; per_device x processes x grad_accum completions per update (memory only)")
ap.add_argument("--lora_r", type=int, default=64)
ap.add_argument("--vllm_mem", type=float, default=0.35)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_completion", type=int, default=128)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--epsilon_high", type=float, default=None)       # DAPO clip-higher; None = symmetric clipping
ap.add_argument("--zoom_tool", default=None, help="frozen model that re-points on zoomed crops (skill-use RL); default: --model")
ap.add_argument("--zoom_cost", type=float, default=0.05)
ap.add_argument("--window_cost", type=float, default=0.05)
ap.add_argument("--base_reward", default="derived_v3", help="per-answer reward inside the skill-use modes (ablations)")
ap.add_argument("--w_loc", type=float, default=0.7, help="share of the readout reward that depends on localization (lambda_loc)")
ap.add_argument("--w_agr", type=float, default=0.2, help="weight of the agreement bonus (lambda_agr)")
ap.add_argument("--w_fmt", type=float, default=0.1, help="reward of a well-formed output (lambda_fmt)")
ap.add_argument("--loc_scale", type=float, default=0.5,
                help="scale r of the localization reward as a fraction of the geometric-mean box side")
ap.add_argument("--cf_credit", type=float, default=0.0, help="counterfactual skill credit: +/- this per skill decision")
ap.add_argument("--cf_margin", type=float, default=0.0, help="a skill call counts as right only if it beats not calling by more than this")
ap.add_argument("--self_tool", action="store_true",
                help="the policy executes its own viewing actions (greedy, through its colocated vLLM engine) instead of a frozen model")
ap.add_argument("--view_data", default=None,
                help="re-pointing prompts (crops, alternative windows; task 'crop') mixed into stage 2, rewarded by localization")
ap.add_argument("--dec_distill", type=float, default=0.0,
                help="weight of the policy-improvement loss on the viewing-decision tokens (target from exact counterfactual values)")
ap.add_argument("--dec_tau", type=float, default=0.1, help="temperature of the improved decision distribution, in reward units")
ap.add_argument("--dec_credit", type=float, default=0.0,
                help="counterfactual viewing credit on the decision tokens (ViewingPolicyGRPO): scale of the per-decision advantage")
ap.add_argument("--dec_values", default=None,
                help="cross-fitted viewing values (xfit_values.py): the zoom decision is credited with the readings, without and "
                     "with zoom, of the Stage-I model fitted on the other half of the patients instead of the policy's own counterfactual")
a = ap.parse_args()
import rewards as _rewards
_rewards.set_reward_weights(a.w_loc, a.w_agr, a.w_fmt, a.loc_scale)

# The vision tower is frozen (LoRA on the language model only), so vLLM's copy loaded from the checkpoint is already
# current. Skip it when syncing weights: transformers 5.17 and vLLM 0.31 name Gemma 3's SigLIP parameters differently.
import trl.generation.vllm_generation as _vg
for _cls in [c for c in vars(_vg).values() if isinstance(c, type) and hasattr(c, "_iter_named_params")]:
    _orig_iter = _cls._iter_named_params
    def _iter_lm_only(self, _orig=_orig_iter):
        for name, param in _orig(self):
            if "vision_tower" not in name and ".visual." not in name and not name.startswith("visual."):
                yield name, param
    _cls._iter_named_params = _iter_lm_only

# TRL re-pads prompts per micro-batch but keeps the processor's mm_token_type_ids, which are aligned to the processor's
# own left padding. When prompts of different lengths share a batch (multi-task data), the types shift against the ids
# and Qwen-VL's 3D rope index fails ("shape mismatch ... [3, 414] ... [3, 407]"). The types are a function of the ids,
# so recompute them from input_ids before every rope-index call.
def _patch_rope_index():
    import importlib
    import torch
    for mod, cls in (("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VLModel"),
                     ("transformers.models.qwen2_vl.modeling_qwen2_vl", "Qwen2VLModel"),
                     ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLModel")):
        try:
            C = getattr(importlib.import_module(mod), cls)
        except (ImportError, AttributeError):
            continue
        if not hasattr(C, "get_rope_index") or getattr(C.get_rope_index, "_patched", False):
            continue
        orig = C.get_rope_index
        def get_rope_index(self, input_ids, mm_token_type_ids=None, *args, _orig=orig, **kw):
            if input_ids is not None:
                cfg = self.config
                t = torch.zeros_like(input_ids, dtype=torch.int)
                t[input_ids == cfg.image_token_id] = 1
                if getattr(cfg, "video_token_id", None) is not None:
                    t[input_ids == cfg.video_token_id] = 2
                mm_token_type_ids = t
            return _orig(self, input_ids, mm_token_type_ids, *args, **kw)
        get_rope_index._patched = True
        C.get_rope_index = get_rope_index


_patch_rope_index()

rows = [json.loads(l) for l in open(a.data) if l.strip()]
if a.view_data:
    rows += [json.loads(l) for l in open(a.view_data) if l.strip()]
ds = Dataset.from_list([{"prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": x["question"]}]}],
                         "image": x["image"], "meta": json.dumps(x)} for x in rows]).cast_column("image", HFImage())


def text_of(c):
    return c[0]["content"] if isinstance(c, list) else c


class Verifier:
    """Frozen counterfactual verifier: does removing the pointed region remove the finding, more than removing matched
    control regions does? N = mean over K controls of sigmoid((drop at point - drop at control) / 0.5)."""
    def __init__(self, name, K=3, r=45):
        import torch
        from transformers import AutoModelForImageTextToText
        sys.path.insert(0, os.environ.get("BATCHREAD_DIR", "vistamed"))   # only for --mode derived_cf (not used here)
        from batchread import read_yesno_batch
        self.read, self.K, self.r, self.cache = read_yesno_batch, K, r, {}
        self.proc = AutoProcessor.from_pretrained(name, min_pixels=448 * 448, max_pixels=448 * 448)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16,
                                                                 device_map=f"cuda:{torch.cuda.current_device()}").eval()

    def necessity(self, items, pts_list):
        import math
        import numpy as np
        from PIL import Image
        from verify import control_point, remove
        out = [0.0] * len(items); jobs = {}
        for i, (it, pts) in enumerate(zip(items, pts_list)):
            if not pts:                                 # no location is claimed, so there is nothing to verify;
                out[i] = 1.0                            # correctness is judged by the derived answer alone
                continue
            q = "Is there a " + ("lung" if it.get("organ") == "lung" else "brain") + " tumour in this image? Answer Yes or No."
            im = Image.open(it["image"]).convert("RGB"); p = (pts[0][0] / it.get("scale", 1.0), pts[0][1] / it.get("scale", 1.0))
            rng = np.random.default_rng(i + 7919 * len(self.cache))
            ims = [remove(im, p, self.r)] + [remove(im, control_point(im, p, self.r, rng), self.r) for _ in range(self.K)]
            if it["image"] not in self.cache:
                ims.append(im)
            jobs.setdefault(q, []).append((i, it["image"], ims))
        for q, js in jobs.items():
            flat = [x for _, _, ims in js for x in ims]
            ms = self.read(self.model, self.proc, flat, q); k = 0
            for i, path, ims in js:
                vals = ms[k:k + len(ims)]; k += len(ims)
                if len(ims) == self.K + 2:
                    self.cache[path] = vals[-1]
                m0 = self.cache[path]; dp = m0 - vals[0]
                if a.nec == "vote":                     # discrete: the pointed region beats at least 2 of 3 controls
                    out[i] = float(sum(dp > (m0 - c) for c in vals[1:self.K + 1]) >= (self.K + 1) // 2)
                else:
                    out[i] = sum(1 / (1 + math.exp(-((dp - (m0 - c)) / 0.5))) for c in vals[1:self.K + 1]) / self.K
        return out


class ZoomTool:
    """The zoom skill as an environment tool for skill-use RL: for completions that end with <zoom>yes</zoom>, crop 2x around
    the first point, let a frozen skill model re-point on the crop (greedy), map the point back. The policy learns only
    the decision (and the coarse point); the tool's re-pointing is fixed during this stage."""
    def __init__(self, name, zoom=2.0, accept=0.35):
        import torch
        from transformers import AutoModelForImageTextToText
        from build_crops import CQ
        self.proc = AutoProcessor.from_pretrained(name, min_pixels=448 * 448, max_pixels=448 * 448)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16,
                                                                 device_map=f"cuda:{torch.cuda.current_device()}").eval()
        self.zoom, self.accept, self.q = zoom, accept, CQ.format(z=zoom)

    def final_texts(self, items, texts):
        import torch
        from PIL import Image
        from eval_zoom import crop_box
        out, jobs = list(texts), []
        for i, (it, t) in enumerate(zip(items, texts)):
            if "<zoom>yes</zoom>" not in t:
                continue
            _, pts, _ = parse(t, 1.0)
            if not pts:
                continue
            x0, y0, w = crop_box(pts[0][0], pts[0][1], self.zoom)
            im = Image.open(it["image"]).convert("RGB").resize((448, 448))
            jobs.append((i, x0, y0, w, im.crop((round(x0), round(y0), round(x0 + w), round(y0 + w))).resize((448, 448), Image.BICUBIC)))
        if not jobs:
            return out
        prompt = self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.q}]}],
                                               add_generation_prompt=True, tokenize=False)
        enc = self.proc(text=[prompt] * len(jobs), images=[[j[4]] for j in jobs], return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=40, do_sample=False)
        dec = self.proc.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (i, x0, y0, w, _), t2 in zip(jobs, dec):
            _, p2, _ = parse(t2, 1.0)
            if p2 and ((p2[0][0] - 224) ** 2 + (p2[0][1] - 224) ** 2) ** 0.5 <= self.accept * 448:
                ans = parse(texts[i], 1.0)[0] or ""; k = w / 448
                out[i] = f"<answer>{ans}</answer><points>[[{round(x0 + p2[0][0] * k)}, {round(y0 + p2[0][1] * k)}]]</points>"
        return out


ZOOM = None


def zoom_rewards(items, texts):
    """Skill-use reward: the derived_v2 reward of the final point (after the zoom skill if the policy called it), minus a
    small cost per call. The decision tag is mandatory in this mode."""
    global ZOOM
    ZOOM = ZOOM or ZoomTool(a.zoom_tool or a.model)
    final = ZOOM.final_texts(items, texts)
    out = []
    for it, t, f in zip(items, texts, final):
        if not STRICT_ZOOM.match(t):
            out.append(0.0); continue
        out.append(reward(it, f if STRICT.match(f) else t, "derived_v2") - a.zoom_cost * ("<zoom>yes</zoom>" in t))
    return out


SKILLS = None
XV = {r["id"]: r for r in (json.loads(l) for l in open(a.dec_values) if l.strip())} if a.dec_values else {}
TRAINER = None      # set after the trainer is built (its vLLM engine executes the viewing actions under --self_tool)
DEC = None


def _vllm_skill_runner():
    from skills import SkillRunner

    class VLLMSkillRunner(SkillRunner):
        """Viewing actions executed by the current policy: greedy re-pointing through the colocated vLLM engine, whose
        weights are synced to the policy at every generation step."""
        def __init__(self, llm, proc):
            super().__init__(None, proc); self.llm = llm

        def _ask(self, ims, questions):
            from vllm import SamplingParams
            texts = [self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
                                                   add_generation_prompt=True, tokenize=False) for q in questions]
            outs = self.llm.generate([{"prompt": t, "multi_modal_data": {"image": im}} for t, im in zip(texts, ims)],
                                     sampling_params=SamplingParams(temperature=0.0, max_tokens=40), use_tqdm=False)
            return [o.outputs[0].text for o in outs]
    return VLLMSkillRunner


def VLLMSkillRunner(llm, proc):
    return _vllm_skill_runner()(llm, proc)


def view_reward(item, text):
    """Re-pointing prompt (a zoomed crop or an alternative window, task 'crop'): point inside the lesion box, weighted by
    centrality, plus format."""
    from rewards import grounding_dense
    _, pts, _ = parse(text, item.get("scale", 1.0))
    if not STRICT.match(text) or not pts:
        return 0.0
    return float(inside(pts[0], item["boxes"][0])) * (0.3 + 0.7 * grounding_dense(item, pts)) + 0.1          # per-completion counterfactual decision values from skill_rewards, read by ViewingPolicyGRPO


def skill_rewards(items, texts):
    """Skill-use reward with two skills (skills.py): derived_v2 reward of the final point after the skills the policy
    called, minus a cost per call. Both decision tags are mandatory. The skills are executed by a frozen copy of the
    stage-1 model (--zoom_tool), so that this stage learns only when to call them."""
    global SKILLS
    if SKILLS is None and a.self_tool:
        SKILLS = VLLMSkillRunner(TRAINER.vllm_generation.llm, AutoProcessor.from_pretrained(a.model, min_pixels=448 * 448, max_pixels=448 * 448))
    if SKILLS is None:
        import torch
        from transformers import AutoModelForImageTextToText
        from skills import SkillRunner
        name = a.zoom_tool or a.model
        proc = AutoProcessor.from_pretrained(name, min_pixels=448 * 448, max_pixels=448 * 448); proc.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map=f"cuda:{torch.cuda.current_device()}").eval()
        SKILLS = SkillRunner(model, proc)
    zf = ["<zoom>yes</zoom>" in t for t in texts]; wf = ["<window>yes</window>" in t for t in texts]
    base = a.base_reward if a.mode == "derived_v3_skills" else "derived_v2"          # e.g. "grounded": stated answer x box
    score = lambda it, t, f: reward(it, f if STRICT.match(f) else t, base)
    final, _ = SKILLS.run(items, texts, zf, wf)
    if a.cf_credit > 0 or a.dec_credit > 0:
        # The skills are deterministic tools, so the outcome of the decision not taken can be computed exactly.
        from skills import ALT
        # with cross-fitted values the zoom alternative is read from the table, so the policy's own flip is not executed
        flip_z = final if XV else SKILLS.run(items, texts, [not z for z in zf], wf)[0]
        flip_w, _ = SKILLS.run(items, texts, zf, [not w for w in wf])
    global DEC
    out, DEC = [], {"zoom": [], "window": []}
    for i, (it, t, f, z, w) in enumerate(zip(items, texts, final, zf, wf)):
        if not STRICT_SKILLS.match(t):
            out.append(0.0); DEC["zoom"].append(None); DEC["window"].append(None); continue
        r = score(it, t, f)
        if a.cf_credit > 0 or a.dec_credit > 0:
            rz, rw = score(it, t, flip_z[i]), score(it, t, flip_w[i])       # window is a no-op on MRI: rw == r
            zt, za = r, rz                                                   # outcome of the zoom decision taken / not taken
            if it["id"] in XV:
                # The model fitted on the other half of the patients reads this slice without and with zoom: the value of
                # zooming on a patient it has not fitted, which the policy's own counterfactual (about zero on patients
                # Stage I has fitted) cannot give.
                yes, no = score(it, *XV[it["id"]]["zoom"]), score(it, *XV[it["id"]]["never"])
                zt, za = (yes, no) if z else (no, yes)
        if a.cf_credit > 0:
            # Reward-level credit: each decision earns +cf_credit when it is the better one, else -cf_credit; calling
            # is the better decision only if it beats not calling by more than cf_margin.
            m = a.cf_margin
            good_z = (zt - za > m) if z else (za - zt <= m)
            good_w = ((r - rw > m) if w else (rw - r <= m)) if it["image"] in ALT else not w
            r += a.cf_credit * ((1 if good_z else -1) + (1 if good_w else -1))
        if a.dec_credit > 0:
            # Decision-level credit (ViewingPolicyGRPO): Q(taken) - Q(alternative), each value being the reward of the
            # final point minus the cost of the call.
            qz = (zt - a.zoom_cost, za) if z else (za - a.zoom_cost, zt)        # (Q(call), Q(no call))
            qw = (r - a.window_cost, rw) if w else (rw - a.window_cost, r)
            DEC["zoom"].append((qz[0] - qz[1]) * (1 if z else -1))
            DEC["window"].append((qw[0] - qw[1]) * (1 if w else -1))
        else:
            DEC["zoom"].append(None); DEC["window"].append(None)
        out.append(r - a.zoom_cost * z - a.window_cost * w)
    return out


def deepeyes_rewards(items, texts):
    """DeepEyes-style tool-use reward (arXiv 2505.14362), reproduced on the same skill model and environment: the zoom
    tool may be called; reward = answer accuracy + format + a tool bonus only when the tool was used and the answer is
    right. The answer is the stated one; the point after the tool is what the model reports as evidence."""
    zf = ["<zoom>yes</zoom>" in t for t in texts]                            # the reward does not read the tool's output
    out = []
    for it, t, z in zip(items, texts, zf):
        # As in DeepEyes, calling the tool is optional: an answer without the decision tags is a trajectory without a
        # call (the tags are mandatory only in our viewing-policy modes).
        if not STRICT.match(t):
            out.append(0.0); continue
        acc = float(answer_ok(it, parse(t, it.get("scale", 1.0))[0]))
        out.append(acc + 0.1 + 0.5 * acc * z)
    return out


SEM = None


def medground_rewards(items, texts):
    """MedGround-R1 spatial-semantic reward (medground.py): answer accuracy + (1 if the point is on the lesion, else
    0.5 x semantic consistency of the pointed region) + format."""
    global SEM
    import torch
    from medground import SemanticScorer
    SEM = SEM or SemanticScorer(f"cuda:{torch.cuda.current_device()}")
    pts = [parse(t, it.get("scale", 1.0))[1] for it, t in zip(items, texts)]
    sem = SEM.scores(items, pts)
    out = []
    for it, t, p, s in zip(items, texts, pts, sem):
        if not STRICT.match(t):
            out.append(0.0); continue
        acc = float(answer_ok(it, parse(t, it.get("scale", 1.0))[0]))
        on = bool(p) and bool(it.get("boxes")) and inside(p[0], it["boxes"][0])
        out.append(acc + (1.0 if on else 0.5 * s) + 0.1)
    return out


def r_window(completions, meta, **kw):
    return [float("<window>yes</window>" in text_of(c)) for c in completions]


def r_zoom(completions, meta, **kw):
    return [float("<zoom>yes</zoom>" in text_of(c)) for c in completions]


VER = None


LABELS = {"laterality_vis": ["left", "right"], "laterality": ["left", "right"], "lobe": list(LOBE_NAMES.values()),
          "segment": list(SEG_NAMES.values()), "quadrant": [f"{s}-{f}" for s in ("left", "right") for f in ("anterior", "posterior")]}


def vote_rewards(items, texts):
    """Label-free consensus (after MedVR's consensus-based credit assignment, arXiv 2604.08203): an answer earns 1 when it
    agrees with the majority answer of the rollouts for the same image. derived_vote votes on the answer derived from
    the points (no label, no box); stated_vote votes on the stated answer, mapped to a canonical label."""
    from collections import Counter
    def ans_of(it, t):
        ans, pts, _ = parse(t, it.get("scale", 1.0))
        if a.mode == "derived_vote":
            return derived_answer(it, pts)
        return next((lab for lab in LABELS[it["task"]] if ans is not None and answer_ok({**it, "answer": lab}, ans)), None)
    got = [ans_of(it, t) if STRICT.match(t) else None for it, t in zip(items, texts)]
    if not getattr(vote_rewards, "logged", False):                  # the local batch must hold whole groups
        print("vote groups:", dict(Counter(it["id"] for it in items)), flush=True); vote_rewards.logged = True
    maj = {}
    for k in {it["id"] for it in items}:
        c = Counter(g for g, it in zip(got, items) if it["id"] == k and g is not None)
        maj[k] = c.most_common(1)[0][0] if c else None
    return [0.0 if not STRICT.match(t) else float(g is not None and g == maj[it["id"]]) + 0.1 for g, it, t in zip(got, items, texts)]


def r_main(completions, meta, **kw):
    global VER
    items = [json.loads(m) for m in meta]
    if a.mode in ("derived_vote", "stated_vote"):
        return vote_rewards(items, [text_of(c) for c in completions])
    if a.mode == "derived_v2_zoom":
        return zoom_rewards(items, [text_of(c) for c in completions])
    if a.mode in ("derived_v2_skills", "derived_v3_skills"):
        texts = [text_of(c) for c in completions]
        full = [i for i, it in enumerate(items) if it["task"] != "crop"]
        if len(full) == len(items):
            return skill_rewards(items, texts)
        global DEC
        out, dec = [0.0] * len(items), {"zoom": [None] * len(items), "window": [None] * len(items)}
        if full:
            r = skill_rewards([items[i] for i in full], [texts[i] for i in full])
            for j, i in enumerate(full):
                out[i] = r[j]; dec["zoom"][i] = DEC["zoom"][j]; dec["window"][i] = DEC["window"][j]
        for i, it in enumerate(items):
            if it["task"] == "crop":
                out[i] = view_reward(it, texts[i])
        DEC = dec
        return out
    if a.mode == "deepeyes":
        return deepeyes_rewards(items, [text_of(c) for c in completions])
    if a.mode == "medground":
        return medground_rewards(items, [text_of(c) for c in completions])
    if a.mode != "derived_cf":
        return [reward(it, text_of(c), a.mode) for c, it in zip(completions, items)]
    VER = VER or Verifier(a.verifier)
    pts = [parse(text_of(c), it.get("scale", 1.0))[1] for c, it in zip(completions, items)]
    N = VER.necessity(items, pts)
    return [reward(it, text_of(c), a.mode, n) for c, it, n in zip(completions, items, N)]


def r_derived(completions, meta, **kw):
    out = []
    for c, m in zip(completions, meta):
        it = json.loads(m); out.append(derived_ok(it, parse(text_of(c), it.get("scale", 1.0))[1]))
    return out


def r_answer(completions, meta, **kw):
    return [float(answer_ok(json.loads(m), parse(text_of(c), json.loads(m).get('scale', 1.0))[0])) for c, m in zip(completions, meta)]


def r_consist(completions, meta, **kw):
    out = []
    for c, m in zip(completions, meta):
        it = json.loads(m); ans, pts, _ = parse(text_of(c), it.get("scale", 1.0)); out.append(consistency(it, ans, pts))
    return out


def r_ground(completions, meta, **kw):
    return [grounding(json.loads(m), parse(text_of(c), json.loads(m).get('scale', 1.0))[1]) for c, m in zip(completions, meta)]


proc = AutoProcessor.from_pretrained(a.model, min_pixels=448 * 448, max_pixels=448 * 448)
cfg = GRPOConfig(
    output_dir=a.out, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=20,
    per_device_train_batch_size=a.per_device, gradient_accumulation_steps=a.grad_accum, num_generations=a.num_gen, max_completion_length=a.max_completion, temperature=a.temperature, epsilon_high=a.epsilon_high,
    beta=a.beta, scale_rewards=a.scale_rewards, bf16=True, gradient_checkpointing=True, logging_steps=5, save_steps=200, save_total_limit=4, report_to=[],
    reward_weights=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem, vllm_max_model_length=4096,
    remove_unused_columns=False, seed=a.seed, dataloader_num_workers=2,
)
lora = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
HOOK = None         # decision targets of the current micro-batch while ViewingPolicyGRPO computes its loss


def _install_decision_hook():
    """Wrap TRL's log-softmax helpers so that, while ViewingPolicyGRPO computes its loss, the log-probabilities of both
    decision words (yes / no) at every viewing-decision position are kept for the policy-improvement loss."""
    import torch
    import trl.trainer.grpo_trainer as G

    def wrap(fn):
        def inner(logits, index, *args, **kw):
            out = fn(logits, index, *args, **kw)
            if HOOK is not None and logits.requires_grad:
                temp = kw.get("temperature", 1.0)                                   # TRL passes it by keyword
                tgt = HOOK["target"][HOOK["offset"]:HOOK["offset"] + logits.size(0)]
                for b, t in (tgt >= 0).nonzero().tolist():
                    lsm = torch.log_softmax(logits[b, t].float() / temp, -1)
                    HOOK["terms"].append((tgt[b, t], lsm[YES_ID], lsm[NO_ID]))
                HOOK["offset"] += logits.size(0)
            return out
        return inner
    G.selective_log_softmax = wrap(G.selective_log_softmax)
    G.selective_log_softmax_and_entropy = wrap(G.selective_log_softmax_and_entropy)


class ViewingPolicyGRPO(GRPOTrainer):
    """GRPO in which each viewing-decision token (the yes/no after <zoom> and <window>) is credited by its own
    counterfactual advantage. The viewing actions are deterministic tools, so the value of the decision not taken is
    computed exactly (skill_rewards). For a binary decision a with alternative a', Q(a) - E_pi[Q] = (1 - pi(a)) (Q(a) -
    Q(a')); the decision token gets this advantage, scaled by dec_credit / std like the sequence advantages, in place of
    the sequence advantage. Unlike the group baseline it does not vanish when every rollout of a prompt makes the same
    decision, and it does not mix the quality of the point into the credit of the decision."""

    def _generate_and_score_completions(self, inputs):
        global DEC
        import torch
        DEC = None
        out = super()._generate_and_score_completions(inputs)
        if (a.dec_credit <= 0 and a.dec_distill <= 0) or DEC is None or len(DEC["zoom"]) != out["completion_ids"].shape[0]:
            return out
        import math
        ids = out["completion_ids"]; B, T = ids.shape
        adv = out["advantages"].float().reshape(B, -1)
        adv = adv.expand(B, T).clone() if adv.shape[1] == 1 else adv.clone()
        target = torch.full((B, T), -1.0, device=adv.device)
        logp = out.get("old_per_token_logps")
        std = (self._metrics["train"]["reward_std"][-1] if self._metrics["train"]["reward_std"] else 1.0) + 1e-4
        tok = self.processing_class.tokenizer; credited, mags = 0, []
        for b in range(B):
            row = ids[b].tolist()
            for tag in ("zoom", "window"):
                d = DEC[tag][b]
                if d is None:
                    continue
                t = next((t for t in range(2, len(row)) if row[t] in DECISION_IDS and tok.decode(row[t - 2:t]) == tag + ">"), None)
                if t is None:
                    continue
                p = float(torch.exp(logp[b, t])) if logp is not None else 0.5
                adv[b, t] = a.dec_credit * (1 - p) * d / std; credited += 1; mags.append(abs(adv[b, t].item()))
                if a.dec_distill > 0:
                    # Policy improvement with exact one-step values: pi*(yes) is proportional to pi(yes) exp(Q(yes) / tau).
                    yes = row[t] == YES_ID
                    p_yes = min(max(p if yes else 1 - p, 1e-6), 1 - 1e-6); dq = d if yes else -d     # Q(yes) - Q(no)
                    target[b, t] = 1 / (1 + (1 - p_yes) / p_yes * math.exp(-max(min(dq / a.dec_tau, 50), -50)))
        self._metrics["train"]["viewing/decisions_credited"].append(credited / max(1, B))
        self._metrics["train"]["viewing/abs_decision_adv"].append(sum(mags) / max(1, len(mags)))
        out["advantages"] = adv
        if a.dec_distill > 0:
            out["dec_target"] = target
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        global HOOK
        import torch
        if a.dec_distill <= 0 or "dec_target" not in inputs:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        HOOK = {"target": inputs["dec_target"], "offset": 0, "terms": []}
        try:
            loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            hook, HOOK = HOOK, None
        if hook["terms"]:
            tgt = torch.stack([x[0] for x in hook["terms"]]).float()
            ly = torch.stack([x[1] for x in hook["terms"]]); ln = torch.stack([x[2] for x in hook["terms"]])
            ce = -(tgt * ly + (1 - tgt) * ln).mean()
            loss = loss + a.dec_distill * ce / self.current_gradient_accumulation_steps
            self._metrics["train"]["viewing/improvement_ce"].append(ce.item())
            self._metrics["train"]["viewing/target_yes"].append(tgt.mean().item())
        return loss


DECISION_IDS = {proc.tokenizer.encode("yes")[0], proc.tokenizer.encode("no")[0]}
YES_ID, NO_ID = proc.tokenizer.encode("yes")[0], proc.tokenizer.encode("no")[0]
if a.dec_distill > 0:
    _install_decision_hook()
trainer = (ViewingPolicyGRPO if a.dec_credit > 0 or a.dec_distill > 0 else GRPOTrainer)(model=a.model, reward_funcs=[r_main, r_answer, r_ground, r_consist, r_derived, r_zoom, r_window], args=cfg, train_dataset=ds,
                      processing_class=proc, peft_config=lora)
TRAINER = trainer
trainer.train()
trainer.save_model(a.out + "/final")
