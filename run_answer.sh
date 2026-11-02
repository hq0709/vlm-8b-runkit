#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=a,b bash run_answer.sh [PORT] : GRPO training (500 steps) of q3v8_cmp_answer, then its evaluation.
# Train log: logs/q3v8_cmp_answer.log   adapter: runs/q3v8_cmp_answer/final   results: results/{big,quad,lobe,liver}_q3v8_cmp_answer.jsonl
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT=${1:-29502}
M=runs/q3v8_sftmt_nocrop/merged
[ -f $M/config.json ] || { echo "missing checkpoint $M"; exit 1; }
rm -rf runs/q3v8_cmp_answer
echo "[$(date +%T)] train q3v8_cmp_answer on GPUs $CUDA_VISIBLE_DEVICES, port $PORT"
$ACC launch --num_processes 2 --main_process_port $PORT train_grpo.py --model $M --data data/med/mt_train_nocrop.jsonl --mode answer --scale_rewards batch --beta 0.08 --lr 1e-5 --temperature 0.8 --epsilon_high 0.28 --per_device 4 --grad_accum 4 --vllm_mem 0.3 --max_steps 500 --seed 0 --out runs/q3v8_cmp_answer > logs/q3v8_cmp_answer.log 2>&1 || { echo "training failed, see logs/q3v8_cmp_answer.log"; exit 1; }
echo "[$(date +%T)] eval q3v8_cmp_answer"
bash eval_all.sh $M runs/q3v8_cmp_answer/final q3v8_cmp_answer nozoom
echo "[$(date +%T)] q3v8_cmp_answer done"
