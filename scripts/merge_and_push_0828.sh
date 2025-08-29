#!/usr/bin/env bash
set -euo pipefail

# Merge LoRA adapters and push merged models to Hugging Face Hub
# Requires: transformers, peft, and HF token with write access.

# -------- Configuration --------
# Expect HF_TOKEN to be provided from the environment (do NOT hardcode secrets here)
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

# Checkpoints
CKPT_1="/home/aiscuser/mindgames/UnstableBaselines/outputs/2025-08-28/04-21-01/ppo-4o-4b-0828-v1/checkpoints/iteration-764"
CKPT_2="/home/aiscuser/mindgames/UnstableBaselines/outputs/2025-08-28/04-21-01/ppo-4o-4b-0828-v1/checkpoints/iteration-400"

# Output dirs for merged artifacts (local)
OUT_1="/home/aiscuser/mindgames/UnstableBaselines/outputs/merged-Qwen3-4B-iter764"
OUT_2="/home/aiscuser/mindgames/UnstableBaselines/outputs/merged-Qwen3-4B-iter400"

# Target HF repos
REPO_1="yinita/mg-ppo-4o-4b-0828-mix-v1-382step"
REPO_2="yinita/mg-ppo-4o-4b-mix-0828-v1-200step"

# -------- Validations --------
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[error] HF_TOKEN is empty. Set it before running." >&2
  exit 1
fi

if [[ ! -d "$CKPT_1" ]]; then
  echo "[error] Missing checkpoint: $CKPT_1" >&2
  exit 1
fi
if [[ ! -d "$CKPT_2" ]]; then
  echo "[error] Missing checkpoint: $CKPT_2" >&2
  exit 1
fi

# -------- Merge & Push: iteration-764 --------
echo "[info] Merging iteration-764 into base: $BASE_MODEL"
python scripts/merge_lora_qwen3.py \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$CKPT_1" \
  --output_dir "$OUT_1" \
  --dtype bfloat16 \
  --device_map auto \
  --push_to_hub \
  --hub_repo "$REPO_1"

echo "[done] Iteration-764 merged and (if authorized) pushed to: $REPO_1"

# -------- Merge & Push: iteration-400 --------
echo "[info] Merging iteration-400 into base: $BASE_MODEL"
python scripts/merge_lora_qwen3.py \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$CKPT_2" \
  --output_dir "$OUT_2" \
  --dtype bfloat16 \
  --device_map auto \
  --push_to_hub \
  --hub_repo "$REPO_2"

echo "[done] Iteration-400 merged and (if authorized) pushed to: $REPO_2"

# -------- Summary --------
echo "[summary] Local merged dirs:"
echo "  - $OUT_1 -> $REPO_1"
echo "  - $OUT_2 -> $REPO_2"
