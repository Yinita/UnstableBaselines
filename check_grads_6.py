#!/usr/bin/env python3
"""
Benchmark Qwen-3 0.6 B on three memory-saving strategies:

  • full   – ordinary forward / backward
  • chunk  – gradient-correct windowed forward / backward
  • ckpt   – activation checkpointing on a % of decoder blocks

Example
-------
python check_vram.py --seq 2048 --chunk 128 --dtype bf16
"""

import argparse, contextlib, time, math, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.checkpoint import checkpoint as checkpoint_wrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, apply_activation_checkpointing
from peft.tuners.lora import LoraLayer      # skip all LoRA modules

QWEN_DEC_LAYER = (
    __import__("transformers.models.qwen3.modeling_qwen3", fromlist=["Qwen3DecoderLayer"])
    .Qwen3DecoderLayer
)

# ─── command-line ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seq",   type=int,   default=2048)
parser.add_argument("--chunk", type=int,   default=512)
parser.add_argument("--dtype", type=str,   default="bf16",
                    choices=["fp32", "fp16", "bf16"])
args = parser.parse_args()

DTYPE  = {"fp32": torch.float32,
          "fp16": torch.float16,
          "bf16": torch.bfloat16}[args.dtype]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL  = "Qwen/Qwen3-0.6B-Base"

torch.manual_seed(0)

# ─── helpers ─────────────────────────────────────────────────────────────
def gpu_mb(): return torch.cuda.max_memory_allocated() / 1e6

@contextlib.contextmanager
def track_vram():
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    yield
    peak = torch.cuda.max_memory_allocated()
    print(f"   VRAM Δpeak: {(peak-start)/1e6:7.1f} MB")

def token_loss(logits, tgt):
    v = logits.size(-1)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, v),
        tgt[:, 1:].reshape(-1),
        reduction="mean"
    )

def build_batch(tok, seq_len):
    text = ["Hello world."*6000]        # dummy sentence
    enc  = tok(text, return_tensors="pt", padding="max_length",
               truncation=True, max_length=seq_len).to(DEVICE)
    return enc["input_ids"]

def make_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, device_map={"": DEVICE}
    ).eval().requires_grad_(True)

def run_full(ids):
    model = make_model()
    torch.cuda.reset_peak_memory_stats()
    t0=time.perf_counter()
    with track_vram():
        out = model(ids, use_cache=False)
        loss = token_loss(out.logits, ids)
        loss.backward()
    t1=time.perf_counter()
    print(f"   loss {loss.item():.6f}   time {t1-t0:.2f}s\n")
    return gpu_mb()

def run_chunk(ids, window):
    model = make_model()
    t0=time.perf_counter()
    with track_vram():
        L = ids.size(1)
        for st in range(0, L-1, window):
            ed = min(st+window, L)
            if st == 0:
                out = model(ids[:, :ed], use_cache=False)
            else:
                # prefix WITH grads (correct gradients)
                prefix_out = model(ids[:, :st], use_cache=True, return_dict=True)
                pkv = prefix_out.past_key_values
                out = model(ids[:, st:ed],
                            past_key_values=pkv,
                            position_ids=torch.arange(st, ed,
                                                      device=DEVICE).unsqueeze(0),
                            use_cache=True, return_dict=True)
            loss = token_loss(out.logits, ids[:, st:ed])
            loss.backward()
    t1=time.perf_counter()
    print(f"   loss {loss.item():.6f}   time {t1-t0:.2f}s\n")
    return gpu_mb()

# activation-ckpt helpers -------------------------------------------------
def make_checkpoint_filter(pct: float):
    def check_fn(m):
        if isinstance(m, LoraLayer):
            return False
        # skip any sub-module that *contains* a LoRA child
        for _n, child in m.named_modules():
            if isinstance(child, LoraLayer):
                return False
        if isinstance(m, QWEN_DEC_LAYER):
            h = (id(m) % 1000) / 1000.0            # ~uniform
            return h < pct
        return False
    return check_fn

def run_ckpt(ids, pct):
    model = make_model()
    pct_txt = int(pct*100)
    # apply checkpointing on-the-fly
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=make_checkpoint_filter(pct)
    )
    model.enable_input_require_grads()

    t0=time.perf_counter()
    with track_vram():
        out  = model(ids, use_cache=False)
        loss = token_loss(out.logits, ids)
        loss.backward()
    t1=time.perf_counter()
    print(f"   loss {loss.item():.6f}   time {t1-t0:.2f}s\n")
    return gpu_mb()

# ─── run all modes ───────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL)
ids = build_batch(tok, args.seq)

print(f"\n🟥  FULL pass (L={args.seq})")
full_mem = run_full(ids)

print(f"\n🟥  FULL pass (L={args.seq})")
full_mem = run_full(ids)
print(f"\n🟥  FULL pass (L={args.seq})")
full_mem = run_full(ids)
print(f"\n🟥  FULL pass (L={args.seq})")
full_mem = run_full(ids)
# print(f"🟧  CHUNK pass window={args.chunk} (grad-correct)")
# chunk_mem = run_chunk(ids, args.chunk)

# for pct in (1.0, 0.8, 0.6, 0.4, 0.2):
for pct in (1.0,): #, 0.8):
    print(f"🟦  CKPT {int(pct*100):>3d}% of blocks")
    ck_mem = run_ckpt(ids, pct)
