#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=a,b bash run_deepeyes.sh [PORT] : GRPO training (500 steps) of q3v8_cmp_deepeyes, then its evaluation.
# Train log: logs/q3v8_cmp_deepeyes.log   adapter: runs/q3v8_cmp_deepeyes/final   results: results/sk_model_*_q3v8_cmp_deepeyes.jsonl
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT=${1:-29503}
M=runs/q3v8_sftmt_nocrop/merged
[ -f $M/config.json ] || { echo "missing checkpoint $M"; exit 1; }
rm -rf runs/q3v8_cmp_deepeyes
echo "[$(date +%T)] train q3v8_cmp_deepeyes on GPUs $CUDA_VISIBLE_DEVICES, port $PORT"
$ACC launch --num_processes 2 --main_process_port $PORT train_grpo.py --model $M --data data/med/mt_train_sq.jsonl --mode deepeyes --scale_rewards batch --beta 0.08 --lr 1e-5 --temperature 0.8 --epsilon_high 0.28 --per_device 4 --grad_accum 4 --vllm_mem 0.3 --max_steps 500 --seed 0 --out runs/q3v8_cmp_deepeyes > logs/q3v8_cmp_deepeyes.log 2>&1 || { echo "training failed, see logs/q3v8_cmp_deepeyes.log"; exit 1; }
echo "[$(date +%T)] eval q3v8_cmp_deepeyes"
bash eval_sk.sh $M runs/q3v8_cmp_deepeyes/final q3v8_cmp_deepeyes model
echo "[$(date +%T)] q3v8_cmp_deepeyes done"
