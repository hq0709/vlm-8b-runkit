#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=a,b bash run_all.sh [PORT] : the three runs one after another on one GPU pair (each run is
# training + evaluation). A failed run does not stop the next one. Driver log: logs/run_all.log
# With 4+ GPUs, run the three scripts in parallel instead, each on its own pair with its own port, e.g.
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh 29501 & CUDA_VISIBLE_DEVICES=2,3 bash run_answer.sh 29502 & ...
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT=${1:-29501}
for s in run_ours.sh run_answer.sh run_deepeyes.sh; do
  echo "[$(date +%T)] start $s" | tee -a logs/run_all.log
  bash $s $PORT 2>&1 | tee -a logs/run_all.log
done
echo "[$(date +%T)] all done" | tee -a logs/run_all.log
