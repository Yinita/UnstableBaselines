import math, torch, torch.nn.functional as F
from typing import Optional
from unstable.core import BaseAlgo


class Reinforce(BaseAlgo):
    def prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        enc  = self.tokenizer([o + a for o, a in zip(obs, acts)], return_tensors="pt", padding=True).to(self.device)
        return enc, advs, obs

    def update(self, steps, scaling: float=1.0, chunk_size: Optional[int]=None):
        enc, advs, obs = self.prepare_batch(steps)
        ids_all, attn_all = enc.input_ids, enc.attention_mask
        B, Lmax = ids_all.shape
        device = self.device
        # prompt length for every sample (no BOS/EOS)
        prompt_lens = torch.tensor([len(self.tokenizer(o, add_special_tokens=False)["input_ids"]) for o in obs], device=device)

        if not chunk_size:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(**enc, use_cache=False)

            logp = F.log_softmax(out.logits, dim=-1) # (B, L, V)
            tgt_ids  = ids_all[:, 1:] # (B, L-1)
            tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)

            # build mask in one vectorised line
            mask = torch.arange(Lmax, device=device).unsqueeze(0) > prompt_lens.unsqueeze(1)
            mask = mask[:, 1:] # discard BOS

            seq_logp = (tok_logp * mask).sum(1) / mask.sum(1).clamp(min=1)
            loss = -(advs * seq_logp).mean() / scaling
            loss.backward()
            return {"loss": loss.item(), "num_steps": len(steps), "mode": "full"}

        # chunked pass
        total_loss_val = 0.0
        for b in range(B):
            ids = ids_all[b, : attn_all[b].sum()].unsqueeze(0) # (1, L_b)
            adv = advs[b:b + 1]
            L = ids.size(1)
            pL = prompt_lens[b].item()
            tgt_total = max(L - pL - 1, 1) # generation tokens

            # walk generation segment in windows of ≤ chunk_size
            for start in range(pL, L - 1, chunk_size):
                end = min(start + chunk_size, L)
                cur = ids[:, start:end] # tokens we want logits for
                tgt = ids[:, start + 1:end] # next-token labels
                if tgt.numel() == 0: break

                # 1) prefix -> forward in no-grad (saves memory)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prefix_out = self.model(ids[:, :start], use_cache=True, return_dict=True)
                pkv = prefix_out.past_key_values

                # 2) current chunk -> forward w/ grad
                posids = torch.arange(start, end, device=device).unsqueeze(0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(cur, past_key_values=pkv, position_ids=posids, use_cache=True, return_dict=True)

                logp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
                tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                seq_logp = tok_logp.mean(1)

                loss_raw = -(adv * seq_logp).mean()
                weight = tgt.size(1) / tgt_total # length-based weight
                loss = (weight * loss_raw) / scaling
                loss.backward()
                total_loss_val += (weight * loss_raw.detach()).item()

        return {"loss": total_loss_val / B, "num_steps": len(steps), "mode": f"chunked({chunk_size})"}





# import math, torch, torch.nn.functional as F
# from unstable.core import BaseAlgo



# class Reinforce(BaseAlgo):

#     def prepare_batch(self, steps):
#         obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
#         advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
#         enc  = self.tokenizer([o + a for o, a in zip(obs, acts)], return_tensors="pt", padding=True).to(self.device)
#         return enc, advs, obs

#     def update(self, steps, scaling: float=1.0, chunk_size: Optional[int]=512):
#         enc, advs, obs = self.prepare_batch(steps)
#         ids_all, attn_all = enc.input_ids, enc.attention_mask
#         B, _ = ids_all.shape
#         prompt_lens = torch.tensor([len(self.tokenizer(o, add_special_tokens=False)["input_ids"]) for o in obs], device=self.device)
#         # self.model.zero_grad(set_to_none=True)
#         total_loss_val = 0.0

#         for b in range(B):
#             ids = ids_all[b, : attn_all[b].sum()].unsqueeze(0) # 1 × L_b
#             adv = advs[b:b+1]
#             L = ids.size(1)
#             pL = prompt_lens[b].item()
#             tgt_total = max(L - pL - 1, 1) # total targets

#             if chunk_size:
#                 # walk the generation part in windows
#                 for start in range(pL, L - 1, chunk_size):
#                     end = min(start + chunk_size, L)
#                     cur = ids[:, start:end]
#                     tgt = ids[:, start+1:end]
#                     if tgt.numel() == 0:
#                         break
#                     # 1) no-grad forward for prefix ( prompt .. start )
#                     with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
#                         prefix_out = self.model(ids[:, :start], use_cache=True, return_dict=True)
#                     pkv = prefix_out.past_key_values # fresh, no grad
#                     posids = torch.arange(start, end, device=self.device).unsqueeze(0)
#                     # 2) grad-enabled forward for current chunk 
#                     with torch.autocast("cuda", dtype=torch.bfloat16):
#                         out = self.model(cur, past_key_values=pkv, position_ids=posids, use_cache=True, return_dict=True)

#                     logp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
#                     tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
#                     seq_logp = tok_logp.mean(1)

#                     loss_raw = -(adv * seq_logp).mean()
#                     weight = tgt.size(1) / tgt_total
#                     loss = (weight * loss_raw) / scaling
#                     loss.backward() # no retain_graph
#                     total_loss_val += (weight * loss_raw.detach()).item()
#             else:


#         return {"loss": total_loss_val / B, "num_steps": len(steps)}

