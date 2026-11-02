#!/usr/bin/env bash
# bash download.sh : fetch the private data bundle and the two starting checkpoints from Hugging Face into this repo.
# Run `hf auth login` first (an account with access to the private repos). No token is stored in this repo.
#   data   -> data/med/..., data/cxr/img448   (then fix_paths.py rewrites the image paths to this repo)
#   models -> runs/q3v8_sftmt_skills/merged, runs/q3v8_sftmt_nocrop/merged
# DL_TMP (default: .download under the repo) holds the tar while it is unpacked; it is deleted afterwards.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=${VENV:-$REPO/.venv}
HF=$VENV/bin/hf; [ -x "$HF" ] || HF=hf
PY=$VENV/bin/python; [ -x "$PY" ] || PY=python3
DL_TMP=${DL_TMP:-$REPO/.download}
mkdir -p "$DL_TMP"
echo "[$(date +%T)] data bundle"
"$HF" download jhq0709/vlm-8b-data --repo-type dataset --local-dir "$DL_TMP"
tar -xf "$DL_TMP/data_8b.tar" -C "$REPO"
rm -rf "$DL_TMP"
"$PY" "$REPO/fix_paths.py" --check
echo "[$(date +%T)] checkpoints"
"$HF" download jhq0709/q3v8-sftmt-skills --local-dir "$REPO/runs/q3v8_sftmt_skills/merged"
"$HF" download jhq0709/q3v8-sftmt-nocrop --local-dir "$REPO/runs/q3v8_sftmt_nocrop/merged"
echo "[$(date +%T)] download done"
