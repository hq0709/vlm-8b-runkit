# vlm-8b-runkit

Three GRPO fine-tuning runs of a Qwen3-VL-8B checkpoint (LoRA, vLLM colocated generation), each followed by its evaluation.

| script | run tag | start checkpoint | training data | evaluation |
|---|---|---|---|---|
| `run_ours.sh` | `q3v8_final` | `runs/q3v8_sftmt_skills/merged` | `data/med/mt_train_sq_augm.jsonl` | `eval_sk.sh`, policies `model never zoom` |
| `run_answer.sh` | `q3v8_cmp_answer` | `runs/q3v8_sftmt_nocrop/merged` | `data/med/mt_train_nocrop.jsonl` | `eval_all.sh ... nozoom` |
| `run_deepeyes.sh` | `q3v8_cmp_deepeyes` | `runs/q3v8_sftmt_nocrop/merged` | `data/med/mt_train_sq.jsonl` | `eval_sk.sh`, policy `model` |

All hyperparameters are fixed inside the scripts (500 steps, seed 0, per-device batch 4, gradient accumulation 4, vLLM memory 0.3). Please do not change them.

## Setup

Python 3.13, NVIDIA driver with CUDA 13 support.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The scripts use `.venv` in the repo root. To use another venv, set `VENV=/path/to/venv`.

## Data and checkpoints

The data and the two starting checkpoints are not part of this repo. They are in private Hugging Face repos. Log in with an account that has access, then download:

```bash
.venv/bin/hf auth login
bash download.sh
```

`download.sh` unpacks the data bundle into `data/`, runs `fix_paths.py --check` (it rewrites the absolute image paths in the jsonl files to this repo's `data/` directory and checks that every referenced file exists), and downloads the checkpoints. No token is stored in the repo.

If you got the bundle some other way, place it as follows and run `python fix_paths.py --check` yourself:

```
data/med/   *.jsonl, alt_window.json, img448/, img448a/, img448w/, lobemaps/, zonemaps/, segmaps/, maps_aug/
data/cxr/img448/
runs/q3v8_sftmt_skills/merged/
runs/q3v8_sftmt_nocrop/merged/
```

## Running

Each run uses two GPUs, given by `CUDA_VISIBLE_DEVICES`, plus a free port for `accelerate`:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh 29501
CUDA_VISIBLE_DEVICES=0,1 bash run_answer.sh 29502
CUDA_VISIBLE_DEVICES=0,1 bash run_deepeyes.sh 29503
```

To run all three one after another on a single GPU pair:

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup bash run_all.sh 29501 > run_all.out 2>&1 &
```

With 4 or more GPUs you can run them in parallel, each on its own pair with its own port:

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup bash run_ours.sh     29501 > ours.out 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 nohup bash run_answer.sh   29502 > answer.out 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 nohup bash run_deepeyes.sh 29503 > deepeyes.out 2>&1 &
```

GPU memory: each training process needs about 62 GB free on an 80 GB GPU (H100/A100 80 GB). Do not share these GPUs with other jobs during training.

Each script trains (`logs/<tag>.log`, adapter in `runs/<tag>/final`), then evaluates on the same two GPUs (`results/*_<tag>.jsonl`, eval logs in `logs/`). A run takes several hours.

## What to send back

- the whole `results/` directory
- the whole `logs/` directory (training logs `logs/q3v8_final.log`, `logs/q3v8_cmp_answer.log`, `logs/q3v8_cmp_deepeyes.log` and the evaluation logs)

Adapters in `runs/<tag>/final` are optional (each a few hundred MB).
