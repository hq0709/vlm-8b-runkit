"""Rewrite absolute image/map paths in the data bundle so that they point into this repo's data/ directory.

The jsonl files (and data/med/alt_window.json) store absolute paths such as <OLD_ROOT>/data/med/img448/x.png or
<OLD_ROOT>/data/cxr/img448/y.png. Every absolute path of the form "<anything>/data/med/..." or "<anything>/data/cxr/..."
is rewritten in place to "<REPO>/data/med/..." / "<REPO>/data/cxr/...". Relative paths (data/med/...) are left as they
are (the run scripts work from the repo root). Running it twice is harmless.

    python fix_paths.py            # rewrite every *.jsonl / *.json under data/
    python fix_paths.py --check    # then verify that every file the runs read exists
"""
import argparse, json, os, re, sys

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "data")
PAT = re.compile(r'"/[^"]*?/data/(med|cxr)/')

# data the three runs read (training sets, evaluation sets, alternative-window map)
NEEDED = ["mt_train_sq_augm.jsonl", "mt_train_nocrop.jsonl", "mt_train_sq.jsonl",
          "px_vis_test_big_sq.jsonl", "px_quad_test_big_sq.jsonl", "lobe_test_sq.jsonl", "liver_test_sq.jsonl",
          "px_vis_test_big.jsonl", "px_quad_test_big.jsonl", "lobe_test.jsonl", "liver_test.jsonl"]


def rewrite(path):
    new_root = '"' + DATA + "/"
    tmp, n = path + ".tmp", 0
    with open(path) as f, open(tmp, "w") as g:
        for line in f:
            line, k = PAT.subn(lambda m: new_root + m.group(1) + "/", line)
            n += k; g.write(line)
    os.replace(tmp, path)
    return n


def check():
    missing, seen = [], 0
    for name in NEEDED:
        p = os.path.join(DATA, "med", name)
        if not os.path.exists(p):
            missing.append(p); continue
        for line in open(p):
            if not line.strip():
                continue
            x = json.loads(line)
            for k in ("image", "lobemap", "segmap"):
                if k in x:
                    seen += 1
                    if not os.path.exists(x[k]):
                        missing.append(x[k])
    alt = os.path.join(DATA, "med", "alt_window.json")
    if not os.path.exists(alt):
        missing.append(alt)
    else:
        for k, v in json.load(open(alt)).items():
            seen += 1
            if not os.path.exists(v):
                missing.append(v)
    print(f"checked {seen} referenced files; missing {len(missing)}")
    for m in missing[:20]:
        print("  missing:", m)
    return not missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="after rewriting, verify that every referenced file exists")
    a = ap.parse_args()
    if not os.path.isdir(DATA):
        sys.exit(f"no data directory at {DATA}")
    total = 0
    for root, _, files in os.walk(DATA):
        for fn in sorted(files):
            if fn.endswith((".jsonl", ".json")):
                n = rewrite(os.path.join(root, fn)); total += n
                if n:
                    print(f"{os.path.relpath(os.path.join(root, fn), REPO)}: {n} paths rewritten")
    print(f"done: {total} paths now under {DATA}/")
    if a.check and not check():
        sys.exit(1)
