import math, torch, torch.nn.functional as F

class Reinforce(BaseAlgo):

    def prepare_batch(self, steps):
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        enc  = self.tokenizer([o + a for o, a in zip(obs, acts)], return_tensors="pt", padding=True).to(self.device)
        return enc, advs, obs

    def update(self, steps, scaling: float=1.0, chunk_size: int=200):
        enc, advs, obs = self.prepare_batch(steps)
        input_ids, attn_mask = enc.input_ids, enc.attention_mask
        B, T = input_ids.shape

        prompt_lens = torch.tensor([len(self.tokenizer(o, add_special_tokens=False)["input_ids"]) for o in obs], device=self.device)

        self.model.zero_grad(set_to_none=True)
        total_loss_val = 0.0

        for b in range(B):
            ids = input_ids[b, : attn_mask[b].sum()].unsqueeze(0)
            adv = advs[b:b+1]
            L = ids.size(1)
            pL = prompt_lens[b].item()

            # 1) build KV cache for prompt (no grad)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(ids[:, :pL], use_cache=True, return_dict=True)
            pkv = out.past_key_values

            # 2) walk remaining tokens in windows 
            tgt_total = L - pL # tokens w/ grad
            num_chunks = math.ceil(tgt_total / chunk_size)

            for start in range(pL, L, chunk_size):
                end = min(start + chunk_size, L)
                cur = ids[:, start:end] # 1 × S
                tgt = ids[:, start+1:end+1] # 1 × S (shifted)
                posids = torch.arange(start, end, device=self.device).unsqueeze(0)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(cur, past_key_values=pkv, position_ids=posids, use_cache=True, return_dict=True)
                pkv = out.past_key_values

                #  loss over this chunk
                logp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
                tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                seq_logp = tok_logp.mean(1) # average over time
                loss_raw = -(adv * seq_logp).mean() # REINFORCE

                # weight so Σ(weight_i) = 1  → same gradient scale
                weight = (end - start) / tgt_total
                loss = (weight * loss_raw) / scaling
                loss.backward()

                total_loss_val += (weight * loss_raw.detach()).item()

        return {"loss": total_loss_val / B, "num_steps": len(steps)}



# import torch 
# from unstable.core import BaseAlgo

# class Reinforce(BaseAlgo):

#     def prepare_batch(self, steps):
#         obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
#         advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
#         enc = self.tokenizer([o + a for o, a in zip(obs, acts)], return_tensors="pt", padding=True).to(self.device)
#         return enc, advs, obs

#     def update(self, steps, scaling: float=1.0, chunk_size: int=200,):
#         enc, advs, obs = self.prepare_batch(steps)
#         input_ids = enc.input_ids
#         attention_mask = enc.attention_mask
#         batch_size, seq_len = input_ids.shape

#         prompt_lens = torch.tensor([len(self.tokenizer(o, add_special_tokens=False)["input_ids"]) for o in obs], device=self.device)

#         # self.model.zero_grad(set_to_none=True) TODO check if necessary

#         ############ 3. iterate over chunks ###########################################
#         # We do it sample-by-sample to keep the code short & clear.  If you need
#         # maximum throughput, vectorise this loop.
#         total_loss = 0.0
#         for b in range(batch_size):
#             ids = input_ids[b, : attention_mask[b].sum()].unsqueeze(0)   # trim padding
#             adv = advs[b : b + 1]
#             prompt_L = prompt_lens[b].item()

#             # 3a. build the KV cache up to the prompt under no-grad
#             with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
#                 out = self.model(ids[:, :prompt_L], use_cache=True, return_dict=True)
#                 pkv = out.past_key_values # list of tuples

#             # 3b. walk through the remainder in `chunk_size` windows
#             for start in range(prompt_L, ids.size(1), chunk_size):
#                 end = min(start + chunk_size, ids.size(1))
#                 cur = ids[:, start:end]                                    # 1 × S
#                 tgt = ids[:, start + 1 : end + 1]                          # 1 × S   (shifted)

#                 # build absolute position ids so rotary/ALiBi stay correct
#                 pos_ids = torch.arange(start, end, device=self.device).unsqueeze(0)

#                 with torch.autocast("cuda", dtype=torch.bfloat16):
#                     out = self.model(cur, past_key_values=pkv, position_ids=pos_ids, use_cache=True, return_dict=True)
#                 pkv = out.past_key_values # extend cache

#                 # -------- loss over this chunk only -----------------------------
#                 logp = torch.nn.functional.log_softmax(out.logits[:, :-1, :], dim=-1)
#                 tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # 1 × S-1
#                 seq_logp = tok_logp.mean(1)                                 # average over time
#                 loss = -(adv * seq_logp).mean() / scaling
#                 loss.backward()                                             # accumulate

#                 total_loss += loss.detach().item()

#         ############ 4. return stats ##################################################
#         return {
#             "loss":       total_loss / batch_size,
#             "logp_mean":  None,   # not tracked here; add if you need
#             "logp_std":   None,
#             "num_steps":  len(steps),
#         }






#     # def update(self, steps, scaling: float = 1.0):
#     #     enc, advs, obs = self.prepare_batch(steps=steps) # unpack
#     #     # out = self.model(**enc) # forward
#     #     with torch.autocast('cuda', dtype=torch.bfloat16):
#     #         out = self.model(**enc, use_cache=False)


#     #     logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
#     #     tgt_ids = enc.input_ids[:, 1:]
#     #     tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
#     #     mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device) # build prompt mask
#     #     for i, o in enumerate(obs):
#     #         L = len(self.tokenizer(o, add_special_tokens=False)["input_ids"])
#     #         mask[i, :L] = False
#     #     mask = mask[:, 1:]
#     #     seq_logp = (tok_logp * mask).sum(1) / mask.sum(1).clamp(min=1)
#     #     loss = -(advs * seq_logp).mean() / scaling
#     #     loss.backward()
#     #     return {"loss": loss.item(), "logp_mean": seq_logp.mean().item(), "logp_std": seq_logp.std().item(), "num_steps": len(steps)}
