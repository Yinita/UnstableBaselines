import torch, torch.nn as nn
from unstable.core import BaseAlgo
from typing import Dict, Any, List, Tuple

class Reinforce(BaseAlgo):

    def prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        combined = [o + a for o, a in zip(obs, acts)]
        lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
        avg_len = sum(lengths) / len(lengths)
        pct_truncated = sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device) # Tokenize with truncation
        return enc, advs, obs, avg_len, pct_truncated

    def update(self, steps, scaling: float = 1.0):
        enc, advs, obs, avg_len, pct_truncated = self.prepare_batch(steps=steps) # unpack
        print(f"TOKENS PER ITEM: {[len(ids) for ids in enc['input_ids']]}")
        out = self.model(**enc) # forward
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device) # build prompt mask
        for i, o in enumerate(obs): mask[i, :len(self.tokenizer(o, add_special_tokens=False)["input_ids"])] = False
        mask = mask[:, 1:]
        seq_logp = (tok_logp * mask).sum(1) / mask.sum(1).clamp(min=1)
        loss = -(advs * seq_logp).mean() / scaling
        loss.backward()
        torch.cuda.empty_cache()
        return {
            "loss": loss.item(), "logp_mean": seq_logp.mean().item(), "logp_std": seq_logp.std().item(),
            "num_steps": len(steps), "avg_train_len": avg_len, "pct_truncated": pct_truncated
        }

class Reinforce2(BaseAlgo):
    """
    Prefix-cached KV-reuse version.
    * self.prefix_len – how many tokens fit on-GPU (int)
    * self.pin_kv_cpu – move KV cache to pinned host RAM (bool)
    """

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _shift_labels(input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing labels: predict next token, ignore last column."""
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        return labels

    @staticmethod
    def _kv_to(kv: List[Tuple[torch.Tensor, torch.Tensor]], device: str):
        """Move a past_key_values list to <device> *without* detaching."""
        return [(k.to(device), v.to(device)) for k, v in kv]

    # ------------------------------------------------------------------ #
    # Batching
    # ------------------------------------------------------------------ #
    def prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)

        combined = [o + a for o, a in zip(obs, acts)]
        lengths  = [len(self.tokenizer(t, add_special_tokens=False)["input_ids"])
                    for t in combined]

        enc = self.tokenizer(
            combined, return_tensors="pt", padding=True,             # ➊ NO truncation
        )                                                            #    keep full ctx

        stats = dict(
            avg_len = sum(lengths) / len(lengths),
            pct_truncated = 0.0                                       # we no longer cut
        )
        return enc, advs, obs, stats

    # ------------------------------------------------------------------ #
    # Single forward pass: prefix on GPU → KV to CPU → tail on GPU
    # ------------------------------------------------------------------ #
    def prefix_cached_forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape
        P    = self.prefix_len

        # Short seq → plain GPU forward
        if S <= P:
            return self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                use_cache=False, return_dict=True
            )

        # 1️⃣ Prefix on GPU
        pfx_ids  = input_ids[:, :P].to(self.device)
        pfx_mask = (attention_mask[:, :P].to(self.device)
                    if attention_mask is not None else None)

        pfx_out  = self.model(pfx_ids, attention_mask=pfx_mask,
                              use_cache=True, return_dict=True)
        kv_cache = pfx_out.past_key_values                             # grads kept

        if self.pin_kv_cpu:
            kv_cache = self._kv_to(kv_cache, "cpu")                    # GPU → CPU

        # 2️⃣ Tail on GPU, feed back cache
        tail_ids  = input_ids[:, P:].to(self.device)
        tail_mask = (attention_mask[:, P:].to(self.device)
                     if attention_mask is not None else None)

        tail_out  = self.model(
            tail_ids,
            attention_mask   = tail_mask,
            past_key_values  = self._kv_to(kv_cache, self.device),
            use_cache        = False,
            return_dict      = True,
        )

        # 3️⃣ Concat logits so loss sees full sequence
        full_logits = torch.cat([pfx_out.logits, tail_out.logits], dim=1)

        class Output:                              # mimic HF output API
            def __init__(self, logits): self.logits = logits
        return Output(full_logits)

    # ------------------------------------------------------------------ #
    # Update step
    # ------------------------------------------------------------------ #
    def update(self, steps, scaling: float = 1.0):
        enc, advs, obs, stats = self.prepare_batch(steps)

        # Forward with prefix-cached KV reuse
        out = self.prefix_cached_forward(enc["input_ids"], enc.get("attention_mask"))

        # Cross-entropy on entire sequence
        logp    = nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc["input_ids"][:, 1:].to(self.device)
        tok_lp  = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

        # Build prompt mask ONCE using cached lengths
        obs_lens = torch.tensor(
            [len(self.tokenizer(o, add_special_tokens=False)["input_ids"])
             for o in obs],
            device=self.device
        )
        mask = torch.arange(enc["input_ids"].size(1) - 1,
                            device=self.device).expand(len(obs), -1)
        mask = mask >= obs_lens.unsqueeze(1)         # False where prompt

        seq_lp = (tok_lp * mask).sum(1) / mask.sum(1).clamp(min=1)
        loss   = -(advs * seq_lp).mean() / scaling

        self.optimizer.zero_grad(set_to_none=True)   # if you keep .optimizer attr
        loss.backward()
        self.optimizer.step()

        torch.cuda.empty_cache()

        # --- telemetry -------------------------------------------------- #
        return {
            "loss"          : loss.item(),
            "logp_mean"     : seq_lp.mean().item(),
            "logp_std"      : seq_lp.std().item(),
            "num_steps"     : len(steps),
            "avg_train_len" : stats["avg_len"],
            "pct_truncated" : stats["pct_truncated"],  # always 0 now
        }


# class Reinforce2(BaseAlgo):

#     def hybrid_forward(self, input_ids, attention_mask=None):
#         batch_size, seq_len = input_ids.shape
        
#         if seq_len <= self.max_train_len: # Short sequences: process entirely on GPU
#             return self.model(input_ids=input_ids, attention_mask=attention_mask)
        
#         # Long sequences: split processing
#         gpu_ids = input_ids[:, :self.max_train_len].to(self.device)
#         cpu_ids = input_ids[:, self.max_train_len:].to('cpu')
        
#         gpu_mask = attention_mask[:, :self.max_train_len].to(self.device) if attention_mask is not None else None
#         cpu_mask = attention_mask[:, self.max_train_len:].to('cpu') if attention_mask is not None else None
        
#         # Process GPU portion
#         with torch.cuda.device(self.device):
#             gpu_out = self.model(input_ids=gpu_ids, attention_mask=gpu_mask)
#             gpu_logits = gpu_out.logits
#             gpu_hidden = gpu_out.last_hidden_state if hasattr(gpu_out, 'last_hidden_state') else None
        
#         # Move model to CPU temporarily for CPU portion
#         model_device = next(self.model.parameters()).device
#         self.model.to('cpu')
        
#         try:
#             # Process CPU portion (with context from GPU portion if available)
#             cpu_out = self.model(input_ids=cpu_ids, attention_mask=cpu_mask)
#             cpu_logits = cpu_out.logits
            
#             # Combine results
#             combined_logits = torch.cat([
#                 gpu_logits.to('cpu'), 
#                 cpu_logits
#             ], dim=1).to(self.device)
            
#         finally:
#             # Move model back to original device
#             self.model.to(model_device)
        
#         # Create output object similar to model output
#         class HybridOutput:
#             def __init__(self, logits):
#                 self.logits = logits
        
#         return HybridOutput(combined_logits)

#     def prepare_batch(self, steps):
#         obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
#         advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
#         combined = [o + a for o, a in zip(obs, acts)]
#         lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
#         avg_len = sum(lengths) / len(lengths)
#         pct_truncated = sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0
        
#         # Don't move to device yet - we'll handle device placement in hybrid_forward
#         enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len)
#         return enc, advs, obs, avg_len, pct_truncated

#     def update(self, steps, scaling: float = 1.0):
#         enc, advs, obs, avg_len, pct_truncated = self.prepare_batch(steps=steps)
#         print(f"TOKENS PER ITEM: {[len(ids) for ids in enc['input_ids']]}")
        
#         # Use hybrid forward pass
#         out = self.hybrid_forward(enc['input_ids'], enc.get('attention_mask'))
        
#         logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
#         tgt_ids = enc.input_ids[:, 1:].to(self.device)
#         tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        
#         # Build prompt mask
#         mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device)
#         for i, o in enumerate(obs): 
#             mask[i, :len(self.tokenizer(o, add_special_tokens=False)["input_ids"])] = False
#         mask = mask[:, 1:]
        
#         seq_logp = (tok_logp * mask).sum(1) / mask.sum(1).clamp(min=1)
#         loss = -(advs * seq_logp).mean() / scaling
#         loss.backward()
#         torch.cuda.empty_cache()
        
#         return {
#             "loss": loss.item(), "logp_mean": seq_logp.mean().item(), "logp_std": seq_logp.std().item(),
#             "num_steps": len(steps), "avg_train_len": avg_len, "pct_truncated": pct_truncated
#         }