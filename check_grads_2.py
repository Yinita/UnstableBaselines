#!/usr/bin/env python3
"""
Compare gradients between:
  • full forward / backward
  • chunked forward / backward (re-runs prefix, identical maths)
The script is deterministic and should show   loss_full == loss_chunk
and   all gradient lines → True.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── configuration ──────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-0.6B-Base"
BATCH      = 1
SEQ_LEN    = 128          # keep small for quick test
CHUNK_SIZE = 32           # >0 enables chunking
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.float32   # use float32 for deterministic norm math
torch.manual_seed(0)
# ────────────────────────────────────────────────────────────────────────────

tok   = AutoTokenizer.from_pretrained(MODEL_ID)
modelF = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
modelC = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

modelF.eval()          # turn off dropout etc.
modelC.eval()
torch.use_deterministic_algorithms(True)

# ---------------------------------------------------------------------------

def make_batch():
    text = ["A quick brown fox jumps over the lazy dog."] * BATCH
    enc  = tok(text,
               return_tensors="pt",
               padding="longest",
               max_length=SEQ_LEN,
               truncation=True).to(DEVICE)
    prompt_lens = torch.full((BATCH,), enc.input_ids.size(1) // 2, device=DEVICE)
    return enc, prompt_lens

def nll_tokens(logits, tgt):
    """token-wise NLL without mean-reduction"""
    vocab = logits.size(-1)
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab),
        tgt[:, 1:].reshape(-1),
        reduction="none"
    ).view(tgt.size(0), -1)                # (B, L-1)

# ---------------------------------------------------------------------------

def full_pass(model, enc, p_lens):
    model.zero_grad(set_to_none=True)
    out = model(**enc, use_cache=False)
    tok_nll = nll_tokens(out.logits, enc.input_ids)           # (B, L-1)

    mask = torch.arange(enc.input_ids.size(1)-1, device=DEVICE).unsqueeze(0) >= p_lens.unsqueeze(1)
    loss_per_seq = (tok_nll * mask).sum(1) / mask.sum(1).clamp(min=1)
    loss = loss_per_seq.mean()
    loss.backward()
    return loss.item(), [p.grad.detach().clone() for p in model.parameters()]

def chunk_pass(model, enc, p_lens):
    model.zero_grad(set_to_none=True)
    ids_all, attn = enc.input_ids, enc.attention_mask
    B, _ = ids_all.shape
    total_loss = 0.0

    for b in range(B):
        ids = ids_all[b, : attn[b].sum()].unsqueeze(0)   # (1, L_b)
        L   = ids.size(1)
        pL  = p_lens[b].item()
        tgt_total = max(L - pL - 1, 1)

        for st in range(pL, L-1, CHUNK_SIZE):
            ed   = min(st + CHUNK_SIZE, L)
            # run full prefix up to ed
            seq  = ids[:, :ed]               # (1, ed)
            with torch.no_grad():            # no dropout noise
                pos_ids = torch.arange(ed, device=DEVICE).unsqueeze(0)
            out  = model(seq, use_cache=False, return_dict=True)
            # out  = model(seq, position_ids=pos_ids, use_cache=False, return_dict=True)

            # compute loss only on tokens st … ed-2  (next-token targets)
            logits_sub = out.logits[:, st:ed-1, :]
            tgt_sub    = ids[:, st+1:ed]
            vocab = logits_sub.size(-1)
            loss_tok = F.cross_entropy(
                logits_sub.reshape(-1, vocab),
                tgt_sub.reshape(-1),
                reduction="none"
            ).view(1, -1)                    # (1, chunk_len-1)

            seq_nll = loss_tok.mean(1)
            weight  = tgt_sub.size(1) / tgt_total
            (weight * seq_nll).mean().backward()
            total_loss += (weight * seq_nll).mean().item()

    return total_loss / B, [p.grad.detach().clone() for p in model.parameters()]

# ---------------------------------------------------------------------------

enc, p_lens = make_batch()

print("• full-pass …")
loss_f, grads_f = full_pass(modelF, enc, p_lens)
print(f"  loss = {loss_f:.6f}")

print(f"• chunked-pass (chunk={CHUNK_SIZE}) …")
loss_c, grads_c = chunk_pass(modelC, enc, p_lens)
print(f"  loss = {loss_c:.6f}")

# ---------------------------------------------------------------------------
print("\nGradient comparison (rtol=1e-3, atol=1e-5)")
mismatch = 0
for name, (g0, g1) in zip(modelF.state_dict().keys(), zip(grads_f, grads_c)):
    same = torch.allclose(g0, g1, rtol=1e-3, atol=1e-5)
    if not same:
        mismatch += 1
    print(f"{name:<50} {same}")

print(f"\nDifferent gradients: {mismatch} / {len(grads_f)}")
