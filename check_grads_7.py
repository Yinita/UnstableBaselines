#!/usr/bin/env python3
"""
Stacked VRAM-saver benchmark on Qwen-3-0.6B.

Examples
--------
# A. Full fp32 baseline (slowest / most memory)
python stacked_vram.py

# B. 100 % ckpt + bf16 + flash attention
python stacked_vram.py --dtype bf16 --flash --ckpt

# C. Minimum footprint (bf16 + flash + no-grad prefix chunk-128)
python stacked_vram.py --dtype bf16 --flash --nograd-prefix 128
"""

import argparse, contextlib, time, math, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.checkpoint import checkpoint as ckpt_wrap
from accelerate.utils import (
    FullyShardedDataParallelCPUOffload,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, apply_activation_checkpointing
from peft.tuners.lora import LoraLayer

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
SEQ_LEN  = 2048
BATCH    = 1

# ─── cmd line ────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
p.add_argument("--flash", action="store_true",
               help="use Flash-Attention-2 via attn_implementation='flash_attention_2'")
p.add_argument("--ckpt", action="store_true",
               help="100 % activation checkpointing")
p.add_argument("--offload", action="store_true",
               help="CPU-offload grads & optimizer via FSDP wrapper")
p.add_argument("--nograd-prefix", type=int, metavar="WINDOW",
               help="chunk with this window size, prefix done under no_grad (loses prefix grads)")
args = p.parse_args()

DTYPE = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[args.dtype]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# ─── memory tracker ──────────────────────────────────────────────────────
@contextlib.contextmanager
def track_vram(tag=""):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated()
    print(f"{tag:>14}  VRAM peak { (peak-start)/1e6:7.1f} MB   "
          f"time {t1-t0:5.2f}s")

# ─── data ----------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
ids = tok(["Hello world."] * BATCH,
          return_tensors="pt",
          padding="max_length",
          truncation=True,
          max_length=SEQ_LEN)["input_ids"].to(DEVICE)

def loss_fn(logits, tgt):
    v = logits.size(-1)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, v), tgt[:, 1:].reshape(-1), reduction="mean")

# ─── build model ---------------------------------------------------------
def load_model():
    kwargs = dict(torch_dtype=DTYPE, device_map={"": DEVICE})
    if args.flash:
        kwargs["attn_implementation"] = "flash_attention_2"
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    mdl.eval().requires_grad_(True)
    if args.offload:
        mdl = FullyShardedDataParallelCPUOffload(mdl)
    return mdl

model = load_model()

# ─── optional: 100 % activation checkpointing ---------------------------
if args.ckpt:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    def filter_all_decoder(m):
        return isinstance(m, Qwen3DecoderLayer) and not any(isinstance(c, LoraLayer)
                                                            for c in m.modules())
    apply_activation_checkpointing(model,
        checkpoint_wrapper_fn=ckpt_wrap, check_fn=filter_all_decoder)
    model.enable_input_require_grads()

# ─── run: either chunk or full ------------------------------------------
def run_full():
    model.zero_grad(set_to_none=True)
    out = model(ids, use_cache=False)
    loss = loss_fn(out.logits, ids)
    loss.backward()

def run_chunk_nomem(window: int):
    model.zero_grad(set_to_none=True)
    L = ids.size(1)
    for st in range(0, L-1, window):
        ed = min(st+window, L)
        if st == 0:
            out = model(ids[:, :ed], use_cache=False)
        else:
            # prefix *without* grads
            with torch.no_grad():
                pkv = model(ids[:, :st], use_cache=True,
                            return_dict=True).past_key_values
            out = model(ids[:, st:ed], past_key_values=pkv,
                        position_ids=torch.arange(st, ed, device=DEVICE).unsqueeze(0),
                        use_cache=True, return_dict=True)
        loss = loss_fn(out.logits, ids[:, st:ed])
        loss.backward()

# ─── warm-up to compile kernels -----------------------------------------
with torch.inference_mode():
    _ = model(ids[:, :64], use_cache=False).logits
torch.cuda.synchronize()

# ─── benchmark -----------------------------------------------------------
if args.nograd_prefix:
    with track_vram("chunk-nograd"):
        run_chunk_nomem(args.nograd_prefix)
else:
    run_tag = "ckpt-100%" if args.ckpt else " full "
    with track_vram(run_tag):
        run_full()
