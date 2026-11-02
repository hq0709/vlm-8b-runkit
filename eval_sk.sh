#!/usr/bin/env bash
# eval_sk.sh MODEL ADAPTER TAG "POLICIES" : reading-skill models on the four *_sq test sets under each skill policy
# (eval_skills.py), on the two GPUs in CUDA_VISIBLE_DEVICES. Results: results/sk_{policy}_{vis,quad,lobe,liver}_TAG.jsonl
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
G0=${CUDA_VISIBLE_DEVICES%%,*}; G1=${CUDA_VISIBLE_DEVICES##*,}
M=$1; AD=""; [ "$2" != "-" ] && AD="--adapter $2"; T=$3; POL=${4:-"model never zoom window both"}
run() { for p in $POL; do CUDA_VISIBLE_DEVICES=$1 $P eval_skills.py --model $M $AD --data data/med/$2_sq.jsonl --policy $p --out results/sk_${p}_$3_$T.jsonl > logs/sk_${p}_$3_$T.log 2>&1; done; }
( run $G0 px_vis_test_big vis; run $G0 lobe_test lobe ) &
( run $G1 px_quad_test_big quad; run $G1 liver_test liver ) &
wait
