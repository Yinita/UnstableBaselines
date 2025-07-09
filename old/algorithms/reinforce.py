import torch
from unstable.core import BaseAlgo



class BaseAlgo:
    def initialize(self, model, tokenizer, device, max_generation_len: int, max_train_len: Optional[int]=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len

    def prepare_batch(self, steps): raise NotImplementedError # Turn a list[Step] into tensors on self.dev. Return whatever update() needs
    def update(self, batch):        raise NotImplementedError # One gradient update on *this worker only*. Must call .backward() but NOT .step(). Return latest loss as float (for logging)




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
        enc, advs, obs, avg_len, pct_truncated = self.prepare_batch(steps=steps)
        out = self.model(**enc)
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device) # build prompt mask
        for i, o in enumerate(obs): mask[i, :len(self.tokenizer(o, add_special_tokens=False)["input_ids"])] = False
        mask = mask[:, 1:]
        seq_logp = (tok_logp * mask).sum(1) / self.max_generation_len
        loss = -(advs * seq_logp).mean() / scaling
        loss.backward()
        torch.cuda.empty_cache()
        return {
            "loss": loss.item(), "logp_mean": seq_logp.mean().item(), "logp_std": seq_logp.std().item(),
            "num_steps": len(steps), "avg_train_len": avg_len, "pct_truncated": pct_truncated
        }
