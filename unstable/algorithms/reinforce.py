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


try:                                   # transformers ≥ 4.39
    from transformers.cache_utils import Cache
except ImportError:                    # older versions fall back to list-of-tuples
    Cache = None

class ReinforceWithOffloading(BaseAlgo):
    @staticmethod
    def _shift_labels(input_ids: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        return labels

    @staticmethod
    def _kv_to(kv, device: str):
        non_blocking = device == "cpu"
        if isinstance(kv, Cache) and hasattr(kv, "to"): return kv.to(device=device, non_blocking=non_blocking)
        if isinstance(kv, (list, tuple)): return [(k.to(device, non_blocking=non_blocking), v.to(device, non_blocking=non_blocking)) for k, v in kv]
        def _map_tensors(obj):
            if torch.is_tensor(obj):            return obj.to(device, non_blocking=non_blocking)
            if isinstance(obj, (list, tuple)):  return type(obj)(_map_tensors(x) for x in obj)
            if isinstance(obj, dict):           return {k: _map_tensors(v) for k, v in obj.items()}
            return obj
        return _map_tensors(kv)

    def prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)

        combined = [o + a for o, a in zip(obs, acts)]
        lengths  = [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in combined]
        enc = self.tokenizer(combined, return_tensors="pt", padding=True)
        stats = dict(avg_len = sum(lengths) / len(lengths), pct_truncated = 0.0)
        return enc, advs, obs, stats

    def prefix_cached_forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape
        P = self.max_train_len
        if S <= P: return self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device) if attention_mask is not None else None, use_cache=False, return_dict=True)
        pfx_ids  = input_ids[:, :P].to(self.device)
        pfx_mask = attention_mask[:, :P].to(self.device) if attention_mask is not None else None
        tail_ids = input_ids[:, P:] # still on CPU for now
        tail_mask = attention_mask[:, P:] if attention_mask is not None else None
        pfx_out = self.model(pfx_ids, attention_mask=pfx_mask, use_cache=True, return_dict=True)
        logits_pfx   = pfx_out.logits
        kv_cache_gpu = pfx_out.past_key_values
        del pfx_out
        torch.cuda.empty_cache()
        if self.pin_kv_cpu:
            kv_cache_cpu = self._kv_to(kv_cache_gpu, "cpu")
            del kv_cache_gpu
            torch.cuda.empty_cache()
            kv_cache_working = kv_cache_cpu
        else:
            kv_cache_working = kv_cache_gpu
        tail_kv = (self._kv_to(kv_cache_working, self.device) if self.pin_kv_cpu else kv_cache_working)
        tail_out = self.model(tail_ids.to(self.device), attention_mask=tail_mask.to(self.device) if tail_mask is not None else None, past_key_values=tail_kv, use_cache=False, return_dict=True)
        full_logits = torch.cat([logits_pfx, tail_out.logits], dim=1)
        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(full_logits)

    def update(self, steps, scaling: float = 1.0):
        enc, advs, obs, stats = self.prepare_batch(steps)
        out = self.prefix_cached_forward(enc["input_ids"], enc.get("attention_mask")) # Forward with prefix-cached KV reuse
        logp = nn.functional.log_softmax(out.logits, dim=-1) # Cross-entropy on entire sequence
        tgt_ids = enc["input_ids"][:, 1:].to(self.device)
        tok_lp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        # Build prompt mask ONCE using cached lengths
        obs_lens = torch.tensor([len(self.tokenizer(o, add_special_tokens=False)["input_ids"]) for o in obs], device=self.device)
        mask = torch.arange(enc["input_ids"].size(1) - 1, device=self.device).expand(len(obs), -1)
        mask = mask >= obs_lens.unsqueeze(1) # False where prompt
        seq_lp = (tok_lp * mask).sum(1) / mask.sum(1).clamp(min=1)
        loss = -(advs * seq_lp).mean() / scaling
        loss.backward()
        torch.cuda.empty_cache()
        return {"loss": loss.item(), "logp_mean": seq_lp.mean().item(), "logp_std": seq_lp.std().item(), "num_steps": len(steps), "avg_train_len": stats["avg_len"], "pct_truncated": stats["pct_truncated"]}

