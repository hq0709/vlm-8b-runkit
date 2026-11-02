#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=a,b bash run_ours.sh [PORT] : GRPO training (500 steps) of q3v8_final, then its evaluation.
# Train log: logs/q3v8_final.log   adapter: runs/q3v8_final/final   results: results/sk_{model,never,zoom}_*_q3v8_final.jsonl
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
PORT=${1:-29501}
M=runs/q3v8_sftmt_skills/merged
[ -f $M/processor_config.json ] || { echo "missing checkpoint $M"; exit 1; }
rm -rf runs/q3v8_final
echo "[$(date +%T)] train q3v8_final on GPUs $CUDA_VISIBLE_DEVICES, port $PORT"
$ACC launch --num_processes 2 --main_process_port $PORT train_grpo.py --model $M --self_tool --data data/med/mt_train_sq_augm.jsonl --mode derived_v3_skills --zoom_cost 0.02 --window_cost 0.02 --base_reward derived_v3 --dec_credit 1.0 --scale_rewards batch --beta 0.08 --lr 1e-5 --temperature 0.8 --epsilon_high 0.28 --per_device 4 --grad_accum 4 --vllm_mem 0.3 --max_steps 500 --seed 0 --out runs/q3v8_final > logs/q3v8_final.log 2>&1
[ -f runs/q3v8_final/final/adapter_model.safetensors ] || { echo "training failed, see logs/q3v8_final.log"; exit 1; }
echo "[$(date +%T)] eval q3v8_final"
bash eval_sk.sh $M runs/q3v8_final/final q3v8_final "model never zoom"
echo "[$(date +%T)] q3v8_final done"
