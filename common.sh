# common.sh: sourced by the run_*.sh and eval_*.sh scripts. Moves to the repo root and sets the environment.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO" || exit 1
VENV=${VENV:-$REPO/.venv}                       # python venv with requirements.txt installed
P=$VENV/bin/python
ACC=$VENV/bin/accelerate
export HF_HOME=${HF_HOME:-$REPO/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
mkdir -p logs results runs
[ -x "$P" ] || { echo "python not found at $P (set VENV=/path/to/venv)"; exit 1; }
case "${CUDA_VISIBLE_DEVICES:-}" in
  *,*) ;;
  *) echo "set CUDA_VISIBLE_DEVICES to exactly two GPU ids, e.g. CUDA_VISIBLE_DEVICES=0,1"; exit 1 ;;
esac
