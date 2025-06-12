#!/usr/bin/env python3
# grad_check_qwen3.py
"""
Verifies that a "chunked" forward/backward (re-running the prefix so it
stays in the autograd graph) gives *identical* gradients to a single
full forward/backward pass, using Qwen3-0.6B.

Torch ≥2.1, Transformers ≥4.40.
"""

import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ───────────────────────── config ────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-0.6B-Base"
BATCH      = 2
SEQ_LEN    = 128          # keep modest so we can run on 12 GB
CHUNK      = 32           # >0 enables chunking
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.float32
RTOL, ATOL = 5e-3, 2e-4   # small fp noise allowance
torch.manual_seed(0)
# ─────────────────────────────────────────────────────────────────

tok   = AutoTokenizer.from_pretrained(MODEL_ID)
m_full = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
m_chunk= AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

for m in (m_full, m_chunk):
    m.eval()            # turn off dropout & rmsnorm noise

# -----------------------------------------------------------------
def batch():
    txt   = ["A quick brown fox jumps over the lazy dog."] * BATCH
    enc   = tok(txt, return_tensors="pt", padding="longest",
                max_length=SEQ_LEN, truncation=True).to(DEVICE)
    # mask out the “prompt” (first half) for RL-style loss
    prompt = torch.full((BATCH,), enc.input_ids.size(1)//2, device=DEVICE)
    return enc, prompt

def token_nll(logits, tgt):
    V = logits.size(-1)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V),
                           tgt[:, 1:].reshape(-1),
                           reduction="none").view(tgt.size(0), -1)

# -----------------------------------------------------------------
def pass_full(model, enc, p_len):
    model.zero_grad(set_to_none=True)
    out   = model(**enc, use_cache=False)
    nll   = token_nll(out.logits, enc.input_ids)
    mask  = torch.arange(enc.input_ids.size(1)-1, device=DEVICE).unsqueeze(0) >= p_len.unsqueeze(1)
    loss  = ((nll * mask).sum(1) / mask.sum(1).clamp(1)).mean()
    loss.backward()
    return loss.item(), [p.grad.clone() for p in model.parameters()]

def pass_chunk(model, enc, p_len):
    model.zero_grad(set_to_none=True)
    ids_all, attn = enc.input_ids, enc.attention_mask
    B, _ = ids_all.shape
    tot_loss = 0.0

    for b in range(B):
        ids = ids_all[b, : attn[b].sum()].unsqueeze(0)   # (1, L_b)
        L   = ids.size(1)
        pL  = p_len[b].item()
        tgt_total = max(L - pL - 1, 1)

        for st in range(pL, L-1, CHUNK):
            ed   = min(st+CHUNK, L)
            seq  = ids[:, :ed]               # prefix + current chunk
            pos  = torch.arange(ed, device=DEVICE).unsqueeze(0)
            out  = model(seq, position_ids=pos, use_cache=False, return_dict=True)

            # predictions at positions st…ed-2  → labels st+1…ed-1
            logits_sub = out.logits[:, st:ed-1]
            tgt_sub = ids[:, st+1:ed]
            V = logits_sub.size(-1)

            loss_tok = F.cross_entropy(logits_sub.reshape(-1, V), tgt_sub.reshape(-1), reduction="none").view(1, -1)

            # weighting so each target token contributes 1/tgt_total
            w = tgt_sub.size(1) / tgt_total
            (w * loss_tok.mean()).backward()

            tot_loss += (w * loss_tok.mean()).item()
            # zero grads that belong to prefix tokens before next chunk
            # so we only accumulate each pair (i,j) once:
            # comment the next line and gradients will mismatch!
            model.zero_grad(set_to_none=False)

    return tot_loss / B, [p.grad.clone() for p in model.parameters()]

# -----------------------------------------------------------------
enc, p_len = batch()

print("• full-pass …")
loss_f, g_f = pass_full(m_full,  enc, p_len)
print(f"  loss = {loss_f:.6f}")

print(f"• chunked-pass (chunk={CHUNK}) …")
loss_c, g_c = pass_chunk(m_chunk, enc, p_len)
print(f"  loss = {loss_c:.6f}")

print("\nGradient comparison (rtol=%.1e atol=%.1e)" % (RTOL, ATOL))
mismatch = 0
for name, (gf, gc) in zip(m_full.state_dict(), zip(g_f, g_c)):
    same = torch.allclose(gf, gc, rtol=RTOL, atol=ATOL)
    print(f"{name:<52} {same}")
    mismatch += 0 if same else 1
print(f"\nDifferent gradients: {mismatch} / {len(g_f)}")
