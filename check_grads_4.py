#!/usr/bin/env python3
"""
Verify that a chunked forward/backward can match full-pass gradients AND
report real GPU-memory usage + runtime.

Usage examples
--------------
# full pass baseline
python grad_check_vram.py --chunk 0

# chunked with 32-token window
python grad_check_vram.py --chunk 32
"""
import argparse, time, contextlib, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── default config ────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
BATCH    = 1
SEQ_LEN  = 128
DTYPE    = torch.float32          # keep fp32 for exactness
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
RTOL, ATOL = 1e-3, 1e-5           # tolerance for grad equality
torch.manual_seed(0)
# ───────────────────────────────────────────────────────────────

# ---------- simple VRAM tracker --------------------------------
@contextlib.contextmanager
def gpu_mem_tracker(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.memory_allocated(device)
        yield
        peak  = torch.cuda.max_memory_allocated(device)
        end   = torch.cuda.memory_allocated(device)
        print(f"   VRAM  Δpeak: {(peak-start)/1e6:7.1f} MB   "
              f"Δnet: {(end-start)/1e6:7.1f} MB")
    else:
        yield

# ---------- helpers -------------------------------------------
def make_batch(tok):
    txt = ["A quick brown fox jumps over the lazy dog."] * BATCH
    enc = tok(txt, return_tensors="pt", padding="longest",
              max_length=SEQ_LEN, truncation=True).to(DEVICE)
    p_lens = torch.full((BATCH,), enc.input_ids.size(1)//2, device=DEVICE)
    return enc, p_lens

def token_nll(logits, tgt):
    V = logits.size(-1)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, V),
        tgt[:, 1:].reshape(-1),
        reduction="none"
    ).view(tgt.size(0), -1)

# ---------- full pass -----------------------------------------
def run_full(model, enc, p_len):
    model.zero_grad(set_to_none=True)
    with gpu_mem_tracker(enc.input_ids.device):
        out  = model(**enc, use_cache=False)
        nll  = token_nll(out.logits, enc.input_ids)
        mask = (torch.arange(enc.input_ids.size(1)-1, device=DEVICE)
                .unsqueeze(0) >= p_len.unsqueeze(1))
        loss = ((nll * mask).sum(1) / mask.sum(1).clamp(1)).mean()
        loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    return loss.item(), grads

# ---------- chunked pass --------------------------------------
def run_chunked(model, enc, p_len, chunk):
    model.zero_grad(set_to_none=True)
    ids_all, attn = enc.input_ids, enc.attention_mask
    B, _ = ids_all.shape
    total_loss = 0.0

    with gpu_mem_tracker(ids_all.device):
        for b in range(B):
            ids = ids_all[b, : attn[b].sum()].unsqueeze(0)
            L   = ids.size(1)
            pL  = p_len[b].item()
            tgt_total = max(L - pL - 1, 1)

            for st in range(pL, L-1, chunk):
                ed  = min(st+chunk, L)
                seq = ids[:, :ed]                           # recomputed prefix
                out = model(seq, use_cache=False, return_dict=True)

                logits = out.logits[:, st:ed-1]
                tgt    = ids[:, st+1:ed]
                V      = logits.size(-1)
                loss_tok = F.cross_entropy(
                    logits.reshape(-1, V), tgt.reshape(-1),
                    reduction="none").view(1, -1)

                w = tgt.size(1) / tgt_total
                (w * loss_tok.mean()).backward()
                total_loss += (w * loss_tok.mean()).item()
    grads = [p.grad.clone() for p in model.parameters()]
    return total_loss / B, grads

def run_chunked_lowmem(model, ids, chunk_size=64, use_grad_prefix=False):
    model.zero_grad(set_to_none=True)
    ids_all, attn = enc.input_ids, enc.attention_mask
    B, _ = ids_all.shape
    total_loss = 0.0

    with gpu_mem_tracker(ids_all.device):
        for b in range(B):
            ids = ids_all[b, : attn[b].sum()].unsqueeze(0)
            L   = ids.size(1)
            pL  = p_len[b].item()
            tgt_total = max(L - pL - 1, 1)
            for st in range(0, ids.size(1)-1, chunk_size):
                ed = min(st+chunk_size, ids.size(1))
                if st == 0:
                    pkv = None
                    chunk = ids[:, st:ed]
                    pos = torch.arange(ed, device=DEVICE).unsqueeze(0)
                else:
                    if use_grad_prefix:
                        prefix_out = model(ids[:, :st], use_cache=True, return_dict=True)
                    else:
                        with torch.no_grad():
                            prefix_out = model(ids[:, :st], use_cache=True, return_dict=True)
                    pkv = prefix_out.past_key_values
                    chunk = ids[:, st:ed]
                    pos = torch.arange(st, ed, device=DEVICE).unsqueeze(0)

                out = model(chunk, past_key_values=pkv, position_ids=pos, use_cache=True, return_dict=True)
                logits = out.logits[:, st:ed-1]
                tgt    = ids[:, st+1:ed]
                V      = logits.size(-1)
                loss_tok = F.cross_entropy(
                    logits.reshape(-1, V), tgt.reshape(-1),
                    reduction="none").view(1, -1)

                w = tgt.size(1) / tgt_total
                (w * loss_tok.mean()).backward()
                total_loss += (w * loss_tok.mean()).item()
    grads = [p.grad.clone() for p in model.parameters()]
    return total_loss / B, grads

# ---------- main ----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=0,
                        help="0 = full pass, N = chunk window")
    args = parser.parse_args()

    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    modelF = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    modelC = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    modelF.eval(); modelC.eval()

    enc, p_len = make_batch(tok)


    input()
    if args.chunk > 0:
        t2 = time.perf_counter()
        print(f"\n▶ CHUNKED pass (window={args.chunk})")
        loss_c, grads_c = run_chunked(modelC, enc, p_len, args.chunk)
        t3 = time.perf_counter()
        print(f"   loss: {loss_c:.6f}   time: {t3-t2:.2f}s")
    else:
        loss_c, grads_c = loss_f, grads_f

    input()
    t0 = time.perf_counter()
    print("\n▶ FULL pass")
    loss_f, grads_f = run_full(modelF, enc, p_len)
    t1 = time.perf_counter()
    print(f"   loss: {loss_f:.6f}   time: {t1-t0:.2f}s")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --------------- gradient comparison -----------------------
    print("\nGradient comparison (rtol=%.1e, atol=%.1e)" % (RTOL, ATOL))
    mismatch = 0
    for name, (g0, g1) in zip(modelF.state_dict(), zip(grads_f, grads_c)):
        same = torch.allclose(g0, g1, rtol=RTOL, atol=ATOL)
        mismatch += 0 if same else 1
        if args.chunk > 0:
            print(f"{name:<52} {same}")
    if args.chunk > 0:
        print(f"\nDifferent gradients: {mismatch} / {len(grads_f)}")
