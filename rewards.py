"""Rewards for answer-and-point RL on the medical suite.

  A  answer correct (0/1)
  B  grounding quality in [0, 1]
       laterality  exp(-(d/2r)^2 / 2) for the first point's distance d to the nodule centre (r = disc radius): 1 at the
                   centre, 0.88 at the disc edge, 0.61 at twice the radius, ~0 far away
       count       F1 of a one-to-one matching of points to lesion boxes (a point matches a box it lies in)
       detect      GT yes: share of points inside a lesion box (0 if none); GT no: 1 if no points, else 0
  Strict format is mandatory: exactly one <answer>...</answer> followed by one <points>[...]</points> and nothing else
  (no repetition, no truncation); anything else scores 0.
  grounded (AND gate)   = A * (0.3 + 0.7 * B) + 0.1     both doors 1.1, right answer alone 0.4, format only 0.1
  answer-only control   = A + 0.1
  grit                  = s_st + s_bf + 1.1 A           GRIT's GRPO-GR: <think>/<rethink> format, a box in the thinking
                                                         (location unchecked, no box label), answer
Evidence-verified modes (no box is ever read):
  C  answer consistent with the model's own points (label-free): laterality -> the point's side of the body midline
     equals the stated side; detect -> "yes" iff at least one point; count -> the number equals the number of points
  consistent            = A * (0.3 + 0.7 * C) + 0.1     image-level answer label + consistency
  selfcons              = C + 0.1                         no label at all
Evidence-derived answering (the stated answer is ignored; the answer is computed from the points by a fixed rule):
  D  derived answer correct: laterality -> side of the first point w.r.t. the body midline; detect -> yes iff points;
     count -> number of points; compared with the image-level label only
  derived               = D + 0.1
  derived_cf            = D * (0.3 + 0.7 * N) + 0.1    N = counterfactual necessity from a frozen verifier (train_grpo.py)
  derived_box           = D * (0.3 + 0.7 * B) + 0.1    when expert boxes exist: the derived answer AND the point on the box
  derived_v2            = D * (0.3 + 0.7 * Bd) + 0.2 F + 0.1   Bd centre-weighted localisation, F stated answer = derived
  derived_v2_nobox      = D + 0.2 F + 0.1
  derived_v2_semi       = derived_v2 on items with a box, derived_v2_nobox on the rest
  derived_v3[_nobox|_semi] = D * (0.3 + 0.7 Bd + 0.2 F) + 0.1 (box) / D * (1 + 0.2 F) + 0.1 (no box): the v2 terms, but
                          faithfulness only on a correct derivation. In v2 a constant stated answer pulled the points
                          toward the side it named (a faithful but wrong policy earned 0.2 + 0.1 on every image).
  Region tasks read the derived answer from an anatomical map at the first point: lobe -> TotalSegmentator lung lobe;
  segment -> TotalSegmentator Couinaud liver segment.
"""
import math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ground import parse, inside

STRICT = re.compile(r"^\s*<answer>[^<]*</answer>\s*<points>\s*\[[^<]*\]\s*</points>\s*(<zoom>(yes|no)</zoom>\s*)?(<window>(yes|no)</window>\s*)?$", re.S)
STRICT_THINK = re.compile(r"^\s*<think>[^<]*</think>\s*<answer>[^<]*</answer>\s*<points>\s*\[[^<]*\]\s*</points>\s*$", re.S)
STRICT_SKILLS = re.compile(r"^\s*<answer>[^<]*</answer>\s*<points>\s*\[[^<]*\]\s*</points>\s*<zoom>(yes|no)</zoom>\s*<window>(yes|no)</window>\s*$", re.S)
# GRIT (GRPO-GR): a <think> pair and then a <rethink> pair, each worth 0.5 when correctly placed; 0.5 when the thinking
# before <rethink> holds at least one syntactically valid bounding box (four comma-separated integers); the location
# of the box is never checked and no box label is read.
GRIT_THINK = re.compile(r"<think>.*?</think>", re.S)
GRIT_RETHINK = re.compile(r"<rethink>.*?</rethink>", re.S)
GRIT_BOX = re.compile(r"\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+")
STRICT_ZOOM = re.compile(r"^\s*<answer>[^<]*</answer>\s*<points>\s*\[[^<]*\]\s*</points>\s*<zoom>(yes|no)</zoom>\s*$", re.S)
ZQ = (" Then add <zoom>yes</zoom> if you want a 2x zoomed-in view around your point to place it more precisely (for example, "
      "when the lesion is small), or <zoom>no</zoom> otherwise.")
WQ = (" Also add <window>yes</window> if you want to see this slice again in a different CT intensity window (soft tissue "
      "for the chest, abdomen for the liver), or <window>no</window> otherwise.")
WP = ("This is the same CT slice shown in a {w} window. Point to the centre of the lesion (the tumour or nodule) in this "
      "image. Answer in exactly this format: <answer>lesion</answer><points>[[x,y]]</points> where (x, y) is in pixels of "
      "this 448 x 448 image, (0, 0) the top-left corner.")


LOBE_NAMES = {1: "right upper", 2: "right middle", 3: "right lower", 4: "left upper", 5: "left lower",
              6: "left middle"}  # 6: chest X-ray lung zone only (build_cxr.py)


def lobe_ok(gt, ans):
    side, level = gt.split()
    other_sides = {"left", "right"} - {side}; other_levels = {"upper", "middle", "lower"} - {level}
    return side in ans and level in ans and not any(w in ans for w in other_sides | other_levels)


SEG_NAMES = {k: f"segment {k}" for k in range(1, 9)}                              # Couinaud segments I-VIII
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}


def seg_nums(ans):
    """The Couinaud segments an answer names (digits 1-8, or roman numerals after "segment")."""
    a = ans.lower()
    nums = {int(x) for x in re.findall(r"(?<![\d.])([1-8])(?![\d.])", a)}
    return nums | {ROMAN[x] for x in re.findall(r"segment\s+(viii|vii|vi|iv|v|iii|ii|i)\b", a)}


def seg_ok(gt, ans):
    return seg_nums(ans) == {int(gt.split()[-1])}


def region_at(item, pt):
    """Anatomical region under a point (point on [0, 1000]) from the item's 448-px region map: the lung lobe (task lobe)
    or the Couinaud liver segment (task segment). A point just outside the organ takes the nearest region within 20 px. On
    chest X-rays the lobe task reads the lung-zone map of the lung on the point's side of the spine midline (nearest zone,
    no distance limit)."""
    import numpy as np
    names = SEG_NAMES if item["task"] == "segment" else LOBE_NAMES
    m = np.load(item["segmap"] if item["task"] == "segment" else item["lobemap"]); x, y = int(round(pt[0] * 0.448)), int(round(pt[1] * 0.448))
    if not (0 <= x < m.shape[1] and 0 <= y < m.shape[0]):
        return None
    if item.get("organ") == "chest_xray":
        # Lung zone on a radiograph: the side comes from the spine midline (as for laterality) and the zone is the nearest
        # zone of that lung, with no distance limit. An effusion or a basal opacity hides the lung edge, so the segmented
        # lung (and the zone map) stops above the lesion; a point on the lesion there still belongs to that lung's zone.
        ids = (1, 2, 3) if x < item["midline"] else (4, 5, 6)
        if m[y, x] in ids:
            return names[int(m[y, x])]
        yy, xx = np.nonzero(np.isin(m, ids))
        if not len(yy):
            return None
        j = np.argmin((yy - y) ** 2 + (xx - x) ** 2)
        return names[int(m[yy[j], xx[j]])]
    if m[y, x]:
        return names[int(m[y, x])]
    yy, xx = np.nonzero(m)
    if not len(yy):
        return None
    j = np.argmin((yy - y) ** 2 + (xx - x) ** 2)
    return names[int(m[yy[j], xx[j]])] if (yy[j] - y) ** 2 + (xx[j] - x) ** 2 <= 400 else None


lobe_at = region_at


def quad_ok(gt, ans):
    side, fb = gt.split("-")
    opp = {"left": "right", "right": "left", "anterior": "posterior", "posterior": "anterior"}
    return side in ans and fb in ans and opp[side] not in ans and opp[fb] not in ans


def answer_ok(item, ans):
    gt, t = item["answer"], item["task"]
    if ans is None:
        return False
    if t == "quadrant":
        return quad_ok(gt, ans)
    if t == "lobe":
        return lobe_ok(gt, ans)
    if t == "segment":
        return seg_ok(gt, ans)
    if t == "count":
        m = re.search(r"\d+", ans); return bool(m) and m.group(0) == gt
    if t in ("laterality", "laterality_vis"):
        return gt in ans and {"left": "right", "right": "left"}[gt] not in ans
    return ans.startswith(gt)


def grounding(item, pts):
    t = item["task"]
    if t == "laterality":
        if not pts:
            return 0.0
        cx, cy, r = item["discs"][0]; d = math.hypot(pts[0][0] - cx, pts[0][1] - cy)
        return math.exp(-0.5 * (d / max(2 * r, 1e-6)) ** 2)
    if t in ("laterality_vis", "quadrant", "lobe", "segment"):
        if not pts or not item.get("boxes"):                 # no box for this item (semi-supervised training)
            return 0.0
        b = item["boxes"][0]
        if inside(pts[0], b):
            return 1.0
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; half = max(math.hypot(b[2] - b[0], b[3] - b[1]) / 2, 1e-6)
        return 0.6 * math.exp(-0.5 * (math.hypot(pts[0][0] - cx, pts[0][1] - cy) / half) ** 2)
    if t == "count":
        used, hit = set(), 0
        for p in pts:
            j = next((j for j, b in enumerate(item["boxes"]) if j not in used and inside(p, b)), None)
            if j is not None:
                used.add(j); hit += 1
        if not pts:
            return 0.0
        prec, rec = hit / len(pts), hit / len(item["boxes"])
        return 0.0 if hit == 0 else 2 * prec * rec / (prec + rec)
    if item["answer"] == "no":
        return 1.0 if not pts else 0.0
    return 0.0 if not pts else sum(any(inside(p, b) for b in item["boxes"]) for p in pts) / len(pts)


LOC_SCALE = 0.5          # r of grounding_dense as a fraction of the geometric-mean box side (set_reward_weights)


def grounding_dense(item, pts):
    """Centre-weighted localisation in [0, 1] for box tasks: exp(-(d / r)^2 / 2), d the first point's distance to the box
    centre, r half the geometric-mean box side (about the lesion radius): 1 at the centre, ~0.6 at the edge, ~0.14 at
    twice the radius. Unlike box membership it still ranks points that are all inside the box."""
    if not pts or not item.get("boxes"):
        return 0.0
    b = item["boxes"][0]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    r = max(LOC_SCALE * math.sqrt(max(b[2] - b[0], 1) * max(b[3] - b[1], 1)), 1e-6)
    return math.exp(-0.5 * (math.hypot(pts[0][0] - cx, pts[0][1] - cy) / r) ** 2)


def faithful(item, ans, pts):
    """The stated answer says what the model's own point implies (the derived answer), right or wrong."""
    d = derived_answer(item, pts)
    return float(d is not None and ans is not None and answer_ok({**item, "answer": d}, ans))


def consistency(item, ans, pts):
    t = item["task"]
    if ans is None:
        return 0.0
    if t in ("laterality", "laterality_vis"):
        if len(pts) != 1 or "midline" not in item:
            return 0.0
        mid = item["midline"] * item.get("scale", 1.0)                     # midline is in 448-px pixels; points are on [0, 1000]
        side = "right" if pts[0][0] < mid else "left"                       # radiological display: image left = patient's right
        return float(side in ans and {"left": "right", "right": "left"}[side] not in ans)
    if t == "count":
        m = re.search(r"\d+", ans); return float(bool(m) and int(m.group(0)) == len(pts))
    yes = ans.startswith("yes"); no = ans.startswith("no")
    return float((yes and len(pts) > 0) or (no and len(pts) == 0))


def derived_answer(item, pts):
    t = item["task"]
    if t in ("lobe", "segment"):
        return region_at(item, pts[0]) if pts else None
    if t == "quadrant":
        if not pts:
            return None
        k = item.get("scale", 1.0)
        return ("right" if pts[0][0] < item["midline"] * k else "left") + "-" + ("anterior" if pts[0][1] < item["ap_center"] * k else "posterior")
    if t in ("laterality", "laterality_vis"):
        if not pts or "midline" not in item:
            return None
        return "right" if pts[0][0] < item["midline"] * item.get("scale", 1.0) else "left"
    if t == "count":
        return str(len(pts))
    return "yes" if pts else "no"


def derived_ok(item, pts):
    d = derived_answer(item, pts)
    return float(d is not None and d == item["answer"])


# Weights of the evidence-derived reward (derived_v3 and its ablations), the paper's lambda_loc, lambda_agr, lambda_fmt:
#   R = D * ((1 - W_LOC) + W_LOC * B_loc + W_AGR * F) + W_FMT        (W_BASE = 1 - W_LOC)
# train_grpo.py sets them from --w_loc, --w_agr, --w_fmt; the defaults are the final configuration.
W_BASE, W_LOC, W_AGR, W_FMT = 0.3, 0.7, 0.2, 0.1


def set_reward_weights(w_loc=0.7, w_agr=0.2, w_fmt=0.1, loc_scale=0.5):
    global W_BASE, W_LOC, W_AGR, W_FMT, LOC_SCALE
    W_BASE, W_LOC, W_AGR, W_FMT, LOC_SCALE = round(1 - w_loc, 10), w_loc, w_agr, w_fmt, loc_scale


def grit_reward(item, text):
    """GRIT's task reward with our answer check: r_format = s_st + s_bf (above), plus r_ans. In GRIT r_ans is a GPT judge
    score plus 0.1 BLEU against the reference answer; our answers are single labels, scored by the rule-based check
    every method here uses, for which BLEU reduces to the same exact match, so r_ans = 1.1 x correct. GRIT's optional
    counting reward does not apply (no counting task)."""
    think = GRIT_THINK.search(text)
    s_st = 0.5 * bool(think) + 0.5 * bool(think and GRIT_RETHINK.search(text, think.end()))
    s_bf = 0.5 * bool(GRIT_BOX.search(text.split("<rethink>", 1)[0]))
    ans, _, _ = parse(text, item.get("scale", 1.0))
    return s_st + s_bf + 1.1 * float(answer_ok(item, ans))


def reward(item, text, mode="grounded", necessity=None):
    if mode == "grit":                                          # GRIT: grounded-reasoning format + answer, no box label
        return grit_reward(item, text)
    if mode == "vigorl":
        if not STRICT_THINK.match(text) or not re.search(r"\(\s*\d+\s*,\s*\d+\s*\)\s*:", text):
            return 0.0
    elif not STRICT.match(text):
        return 0.0
    ans, pts, _ = parse(text, item.get("scale", 1.0))
    a = float(answer_ok(item, ans))
    if mode == "vigorl":                                        # ViGoRL: accuracy + format, with a grounded <think> trace
        return float(answer_ok(item, ans)) + 0.1
    if mode == "answer":                                        # Med-R1 / MedVLM-R1: accuracy (+ format)
        return a + 0.1
    if mode == "additive":                                      # Visual-RFT / VLM-R1 / Perception-R1: accuracy + localisation
        return a + grounding(item, pts) + 0.1
    if mode == "derived":
        return derived_ok(item, pts) + 0.1
    if mode == "derived_box":
        return derived_ok(item, pts) * (0.3 + 0.7 * grounding(item, pts)) + 0.1
    if mode == "derived_v2":                                    # derived answer x dense evidence + faithful statement
        return derived_ok(item, pts) * (0.3 + 0.7 * grounding_dense(item, pts)) + 0.2 * faithful(item, ans, pts) + 0.1
    if mode == "derived_v2_nobox":                              # the same without boxes
        return derived_ok(item, pts) + 0.2 * faithful(item, ans, pts) + 0.1
    if mode in ("derived_v3", "derived_v3_semi") and item.get("boxes"):    # faithfulness counts only on a correct derivation
        d = derived_ok(item, pts)
        return d * (W_BASE + W_LOC * grounding_dense(item, pts) + W_AGR * faithful(item, ans, pts)) + W_FMT
    if mode == "derived_v3_nof":                                # ablation: no faithfulness term
        return derived_ok(item, pts) * (W_BASE + W_LOC * grounding_dense(item, pts)) + W_FMT
    if mode in ("derived_v3_nobox", "derived_v3_semi"):
        d = derived_ok(item, pts)
        return d * (1 + W_AGR * faithful(item, ans, pts)) + W_FMT
    if mode == "derived_v2_semi":                               # boxes where they exist, answer labels everywhere
        if item.get("boxes"):
            return derived_ok(item, pts) * (0.3 + 0.7 * grounding_dense(item, pts)) + 0.2 * faithful(item, ans, pts) + 0.1
        return derived_ok(item, pts) + 0.2 * faithful(item, ans, pts) + 0.1
    if mode == "derived_cf":
        return derived_ok(item, pts) * (0.3 + 0.7 * (necessity if necessity is not None else 0.0)) + 0.1
    if mode == "consistent":
        return a * (0.3 + 0.7 * consistency(item, ans, pts)) + 0.1
    if mode == "selfcons":
        return consistency(item, ans, pts) + 0.1
    return a * (0.3 + 0.7 * grounding(item, pts)) + 0.1
