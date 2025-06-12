#!/usr/bin/env python3
# grad_check_qwen3.py
"""
Compare gradients between:
  • a full forward / backward pass
  • a chunked pass (prefix in or out of autograd graph)

Change these knobs to experiment:
  detach_prefix  – True reproduces your current code
  chunk_size     – any positive int enables chunking
  seq_len        – keep small for low-VRAM cards
"""

import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.checkpoint import checkpoint

MODEL_ID      = "Qwen/Qwen3-0.6B-Base"
BATCH         = 2
SEQ_LEN       = 128
CHUNK_SIZE    = 32          # >0 → chunked pass
DETACH_PREFIX = False #True        # False recomputes prefix w/ grad

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch_dtype=torch.float32 #torch.bfloat16 if device == "cuda" else torch.float32
torch.manual_seed(0)

tok  = AutoTokenizer.from_pretrained(MODEL_ID)
mdlF = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
mdlC = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
mdlC.gradient_checkpointing_enable()
mdlC.enable_input_require_grads()
# ---------------------------------------------------------------------
def make_batch():
    text = ["A quick brown fox jumps over the lazy dog."] * BATCH
    enc  = tok(text,
               return_tensors="pt",
               padding="longest",
               max_length=SEQ_LEN,
               truncation=True).to(device)
    # pretend first half of each sequence is "prompt"
    prompt_lens = torch.full((BATCH,), enc.input_ids.size(1)//2, device=device)
    return enc, prompt_lens

def gen_nll(logits, tgt_ids):
    vocab = logits.size(-1)
    return F.cross_entropy(logits[:, :-1, :].reshape(-1, vocab),
                           tgt_ids[:, 1:].reshape(-1),
                           reduction="none").view(tgt_ids.size(0), -1)

# ------------------------------------------------------------------ #
# full pass
def full_pass(model, enc, p_lens):
    model.zero_grad(set_to_none=True)
    out  = model(**enc, use_cache=False)
    tok_nll = gen_nll(out.logits, enc.input_ids)     # (B, L-1)
    # mask prompt tokens
    mask = torch.arange(enc.input_ids.size(1)-1,
                        device=device).unsqueeze(0) >= p_lens.unsqueeze(1)
    nll  = (tok_nll * mask).sum(1) / mask.sum(1).clamp(min=1)
    loss = nll.mean()
    loss.backward()
    return loss.item(), [p.grad.detach().clone() for p in model.parameters()]

# --- replace the old chunk_pass with this ---------------------------------
def chunk_pass(model, enc, p_lens):
    model.zero_grad(set_to_none=True)
    ids_all, attn_all = enc.input_ids, enc.attention_mask
    B, _ = ids_all.shape
    total_loss = 0.0

    for b in range(B):
        ids = ids_all[b, : attn_all[b].sum()].unsqueeze(0)   # (1, L_b)
        L   = ids.size(1)
        pL  = p_lens[b].item()
        tgt_total = max(L - pL - 1, 1)

        # walk the generation part
        for st in range(pL, L - 1, CHUNK_SIZE):
            ed   = min(st + CHUNK_SIZE, L)
            cur  = ids[:, st:ed]          # tokens we feed
            tgt  = ids[:, st + 1:ed]      # next-token labels (len = len(cur)-1)
            if tgt.numel() == 0:
                break

            # ----- prefix KV ------------------------------------------------
            if DETACH_PREFIX:
                with torch.no_grad():
                    pkv = model(ids[:, :st], use_cache=True,
                                return_dict=True).past_key_values
            else:
                # pkv = checkpoint(
                #     lambda x: model(x, use_cache=True, return_dict=True),
                #     ids[:, :st]).past_key_values
                def prefix_fn(dummy: torch.Tensor):
                    # we "close over" ids[:, :st] inside the function
                    return model(ids[:, :st], use_cache=True, return_dict=True)

                dummy = torch.ones(1, requires_grad=True, device=device)
                pkv = checkpoint(prefix_fn, dummy).past_key_values

            # ----- forward chunk with grad ----------------------------------
            pos  = torch.arange(st, ed, device=device).unsqueeze(0)
            out  = model(cur, past_key_values=pkv, position_ids=pos, use_cache=True, return_dict=True)
            # pos  = torch.arange(st, ed, device=device).unsqueeze(0)
            # out  = model(cur, past_key_values=pkv, use_cache=True, return_dict=True)

            # logits for every token in `cur`, but we only have targets
            # for positions st..ed-2  (len = len(cur)-1)
            logits = out.logits[:, :-1, :]     # (1, len(cur)-1, V)
            vocab  = logits.size(-1)
            loss_tok = F.cross_entropy(
                logits.reshape(-1, vocab),
                tgt.reshape(-1),
                reduction="none"
            ).view(1, -1)                      # (1, len(cur)-1)

            seq_nll = loss_tok.mean(1)         # average over this chunk
            w       = tgt.size(1) / tgt_total  # length-based weight
            (w * seq_nll).mean().backward()
            total_loss += (w * seq_nll).mean().item()

    return total_loss / B, [p.grad.detach().clone() for p in model.parameters()]
# --------------------------------------------------------------------------


# ------------------------------------------------------------------ #
enc, p_lens = make_batch()
mdlF.eval()
mdlC.eval()
# torch.use_deterministic_algorithms(True)
print("• full-pass …")
loss_full,  grads_full  = full_pass(mdlF, enc, p_lens)
print(f"  loss = {loss_full:.6f}")

print(f"• chunked-pass  (chunk={CHUNK_SIZE}, detach_prefix={DETACH_PREFIX}) …")
loss_chunk, grads_chunk = chunk_pass(mdlC, enc, p_lens)
print(f"  loss = {loss_chunk:.6f}")

print("\nGradient comparison (rtol=1e-3, atol=1e-5)")
mismatch = 0
for name, (g0, g1) in zip(mdlF.state_dict().keys(), zip(grads_full, grads_chunk)):
    same = torch.allclose(g0, g1, rtol=1e-3, atol=1e-5)
    if not same:
        mismatch += 1
    print(f"{name:<50} {same}")
print(f"\nDifferent grads: {mismatch} / {len(grads_full)}")
