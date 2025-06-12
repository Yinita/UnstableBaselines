#!/usr/bin/env python3
"""
Benchmark Qwen-3 0.6 B with several VRAM-saving strategies:

  • full   – ordinary forward/backward
  • chunk  – gradient-correct windowed forward/backward
  • ckpt   – activation-checkpointing on 100 % or a % of blocks
  • offload (optional) – CPU-offload grads/optimizer via FSDP

Examples
--------
# Default: bf16 + 100 % activation-ckpt
python check_vram.py --seq 4096

# bf16 + 100 % ckpt + CPU-offload
python check_vram.py --seq 4096 --offload

# Disable ckpt (fp32 baseline)
python check_vram.py --dtype fp32 --no-ckpt

# Gradient-correct chunking, window 512
python check_vram.py --chunk 512 --no-ckpt
"""
import argparse, contextlib, time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- torch-distributed checkpoint utils -------------------------------
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
)
from torch.utils.checkpoint import checkpoint   # plain checkpoint helper

# ---- FSDP for optional CPU-offload ------------------------------------
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import enable_wrap, wrap

# ---- skip all LoRA layers for ckpt filter -----------------------------
from peft.tuners.lora import LoraLayer

# ---- Qwen3 decoder block class ----------------------------------------
QWEN_DEC_LAYER = (
    __import__("transformers.models.qwen3.modeling_qwen3",
               fromlist=["Qwen3DecoderLayer"]).Qwen3DecoderLayer
)

# ─── CLI ----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seq",   type=int, default=4096,
                    help="sequence length")
parser.add_argument("--chunk", type=int, default=512,
                    help="window size for gradient-correct chunk test")
parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"],
                    default="bf16", help="model precision")
parser.add_argument("--offload", action="store_true",
                    help="CPU-offload grads & optimizer via FSDP")
parser.add_argument("--no-ckpt", action="store_true",
                    help="disable activation checkpointing (baseline)")
args = parser.parse_args()

DTYPE = dict(fp32=torch.float32,
             fp16=torch.float16,
             bf16=torch.bfloat16)[args.dtype]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL  = "Qwen/Qwen3-4B-Base"

torch.manual_seed(0)

# ─── helpers ------------------------------------------------------------
def gpu_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1e6

@contextlib.contextmanager
def track_vram(tag: str = ""):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated()
    print(f"{tag:>12}  Δpeak { (peak-start)/1e6:7.1f} MB   "
          f"time {t1-t0:5.2f}s")

def token_loss(logits, tgt):
    v = logits.size(-1)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, v),
        tgt[:, 1:].reshape(-1),
        reduction="mean"
    )

def build_batch(tokenizer, seq_len: int) -> torch.Tensor:
    txt = ["Hello world."*5000]  # dummy text
    enc = tokenizer(
        txt, return_tensors="pt", padding="max_length",
        truncation=True, max_length=seq_len
    ).to(DEVICE)
    return enc["input_ids"]

# ---- model loader with ckpt & offload switches -------------------------
def make_model():
    kwargs = dict(torch_dtype=DTYPE, device_map={"": DEVICE})
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
    model.eval().requires_grad_(True)

    # 100 % activation-checkpointing (unless --no-ckpt)
    if not args.no_ckpt:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda m: isinstance(m, QWEN_DEC_LAYER)
                               and not any(isinstance(c, LoraLayer)
                                           for c in m.modules()),
        )
        model.enable_input_require_grads()

    # optional FSDP CPU-offload
    if args.offload:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl",
                                                 rank=0, world_size=1)
        with enable_wrap(wrapper_cls=FSDP,
                         cpu_offload=True, mixed_precision=True):
            model = wrap(model)

    return model.to(DEVICE)

# ─── test modes ---------------------------------------------------------
def run_full(ids):
    # model = make_model()
    kwargs = dict(torch_dtype=DTYPE, device_map={"": DEVICE})
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
    model.eval().requires_grad_(True)
    with track_vram("FULL"):
        out = model(ids, use_cache=False)
        loss = token_loss(out.logits, ids)
        loss.backward()

def run_chunk(ids, window: int):
    model = make_model()
    L = ids.size(1)
    with track_vram("CHUNK"):
        for st in range(0, L-1, window):
            ed = min(st+window, L)
            if st == 0:
                out = model(ids[:, :ed], use_cache=False)
            else:
                # prefix WITH grads (correct)
                prefix_out = model(ids[:, :st], use_cache=True,
                                   return_dict=True)
                pkv = prefix_out.past_key_values
                out = model(ids[:, st:ed],
                            past_key_values=pkv,
                            position_ids=torch.arange(st, ed,
                                                      device=DEVICE)
                                          .unsqueeze(0),
                            use_cache=True, return_dict=True)
            loss = token_loss(out.logits, ids[:, st:ed])
            loss.backward()

def make_ckpt_filter(pct: float):
    def fn(m):
        if isinstance(m, LoraLayer):
            return False
        if isinstance(m, QWEN_DEC_LAYER):
            return (id(m) % 1000) / 1000.0 < pct
        return False
    return fn

def run_ckpt(ids, pct: float):
    model = make_model()
    # re-patch ckpt percentage on-the-fly
    model.enable_input_require_grads()
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=make_ckpt_filter(pct)
    )

    with track_vram(f"CKPT {int(pct*100):3d}%"):
        out = model(ids, use_cache=False)
        loss = token_loss(out.logits, ids)
        loss.backward()

# ─── driver -------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL)
ids = build_batch(tok, args.seq)

print(f"\n── Running on {DEVICE} | dtype {args.dtype} | "
      f"seq {args.seq} | offload {args.offload} ──")

run_full(ids)

# print(f"\n── Chunk window {args.chunk} ──")
# run_chunk(ids, args.chunk)

if args.no_ckpt:
    print("\n(activation checkpointing disabled)")
else:
    for pct in (1.0, 0.8, 0.6, 0.4, 0.2):
        run_ckpt(ids, pct)
