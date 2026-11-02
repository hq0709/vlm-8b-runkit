#!/usr/bin/env bash
# eval_all.sh MODEL ADAPTER TAG [nozoom] : single-pass and zoom-and-re-point evaluation on all four test sets, on the two
# GPUs in CUDA_VISIBLE_DEVICES. ADAPTER may be "-" for none. Results: results/{big,quad,lobe,liver}_TAG.jsonl
# and (without nozoom) results/zoom_{vis,quad,lobe,liver}_TAG.jsonl
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
G0=${CUDA_VISIBLE_DEVICES%%,*}; G1=${CUDA_VISIBLE_DEVICES##*,}
M=$1; AD=""; [ "$2" != "-" ] && AD="--adapter $2"; T=$3
e() { CUDA_VISIBLE_DEVICES=$1 $P eval_ground.py --model $M $AD --data data/med/$2.jsonl --out results/$3_$T.jsonl > logs/$3_$T.log 2>&1; }
z() { CUDA_VISIBLE_DEVICES=$1 $P eval_zoom.py --model $M $AD --data data/med/$2.jsonl --zoom 2 --accept 0.25 --out results/zoom_$3_$T.jsonl > logs/zoom_$3_$T.log 2>&1; }
( e $G0 px_vis_test_big big; e $G0 lobe_test lobe; [ "$4" = nozoom ] || { z $G0 px_vis_test_big vis; z $G0 lobe_test lobe; } ) &
( e $G1 px_quad_test_big quad; e $G1 liver_test liver; [ "$4" = nozoom ] || { z $G1 px_quad_test_big quad; z $G1 liver_test liver; } ) &
wait
