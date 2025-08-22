#!/usr/bin/env python3
"""
Merge a PEFT LoRA adapter (iteration checkpoint) into the base Qwen model weights, and save a standalone merged model.

Example:
  python scripts/merge_lora_qwen3.py \
    --base_model Qwen/Qwen3-8B \
    --adapter_dir /home/aiscuser/mindgames/UnstableBaselines/outputs/2025-08-22/00-24-49/MixedPlay-Qwen3-8B-Codenames-v0-train-1755822283/checkpoints/iteration-456 \
    --output_dir /home/aiscuser/mindgames/UnstableBaselines/outputs/2025-08-22/00-24-49/merged-Qwen3-8B-iter456

Notes:
- Requires `transformers` and `peft` installed.
- This script loads the base model + LoRA adapter, calls `merge_and_unload()`, and saves the merged model weights and tokenizer.
- By default loads in bfloat16 if available, otherwise float16; override with --dtype.
"""
from __future__ import annotations
import argparse
import os
import sys

from typing import Optional


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into Qwen/Qwen3-8B base model")
    ap.add_argument("--base_model", default="Qwen/Qwen3-8B", help="HF model id or local path for base model")
    ap.add_argument("--adapter_dir", required=True, help="Path to PEFT LoRA adapter dir (e.g., checkpoints/iteration-456)")
    ap.add_argument("--output_dir", required=True, help="Where to save the merged model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Load dtype")
    ap.add_argument("--device_map", default="auto", help="Device map for loading (e.g., 'auto', 'cpu', 'cuda:0')")
    ap.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to HF loaders")
    ap.add_argument("--push_to_hub", action="store_true", help="If set, push merged model to the Hub (configure HF token)")
    ap.add_argument("--hub_repo", default="", help="Target repo name when pushing to Hub")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Lazy imports to avoid import cost when showing help
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    trust = bool(args.trust_remote_code)

    print(f"[info] Loading base model: {args.base_model} ({dtype}) device_map={args.device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=trust,
    )

    print(f"[info] Loading tokenizer for: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=trust,
        use_fast=False,
    )

    print(f"[info] Attaching PEFT adapter from: {args.adapter_dir}")
    peft_model = PeftModel.from_pretrained(model, args.adapter_dir, torch_dtype=dtype)

    print("[info] Merging LoRA weights into base (merge_and_unload)...")
    merged = peft_model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[info] Saving merged model to: {args.output_dir}")
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # Optional: push to hub
    if args.push_to_hub:
        target_repo = args.hub_repo.strip() or None
        print(f"[info] Pushing merged model to Hub repo: {target_repo or '(use default)'}")
        merged.push_to_hub(target_repo) if target_repo else merged.push_to_hub()
        tokenizer.push_to_hub(target_repo) if target_repo else tokenizer.push_to_hub()

    print("[done] Merge complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
