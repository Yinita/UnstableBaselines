
import time, pathlib, math, json
from typing import Any, Dict, Optional, List

import ray, torch, transformers

# local imports
from unstable.buffer import StepBuffer
from unstable.core import BaseAlgo
from unstable.model_pool import ModelPool
from unstable.learners.utils import  make_checkpointing_filter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig


import deepspeed


def build_peft_model(base_name: str, lora_cfg: Dict[str, Any] | None, freeze_base: bool = True):
    print(f"[Learner] Loading base model: {base_name} ...")
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16, trust_remote_code=True)

    if freeze_base:
        for p in base.parameters(): p.requires_grad_(False)
    
    lcfg = LoraConfig(
        r=lora_cfg.get("lora_rank", 32), lora_alpha=lora_cfg.get("lora_alpha", 32), lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none", task_type="CAUSAL_LM", target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    model = get_peft_model(base, lcfg) 
    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token  # safe default
    return model, tok

def prepare_batch(self, steps):
    obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in steps])
    advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
    enc = tokenizer([o + a for o, a in zip(obs, acts)], return_tensors="pt", padding=True).to(self.device)
    return enc, advs, obs

def train_func(config):
    step_buffer = config["step_buffer"]
    tracker = config["tracker"]
    model_pool = config["model_pool"]

    model, tokenizer = build_peft_model(base_name=config["model_name"], lora_cfg=config["lora_cfg"]) # Instantiate the model
    model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config["deepspeed_config"]) # TODO only pass trainable

    _sample_seen = 0

    for iteration in range(config["iterations"]):

        # wait until batch is ready
        while (ray.get(step_buffer.size.remote()) < config["batch_size"] * 1.5): 
            time.sleep(0.2)

        batch: List = ray.get(step_buffer.get_batch.remote(config["batch_size"]))
        _sample_seen += len(batch)

        optimizer.zero_grad()

        # pass over data 
        obs, acts, advs = zip(*[(s.obs, s.act, s.reward) for s in batch])
        advs = torch.tensor(advs, dtype=torch.float32) #, device=self.device)
        enc = tokenizer([o + a for o, a in zip(obs, acts)], return_tensors="pt", padding=True)

        out = model(**enc) # forward
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool) #, device=self.device) # build prompt mask
        for i, o in enumerate(obs):
            L = len(tokenizer(o, add_special_tokens=False)["input_ids"])
            mask[i, :L] = False
        mask = mask[:, 1:]
        seq_logp = (tok_logp * mask).sum(1) / mask.sum(1).clamp(min=1)
        loss = -(advs * seq_logp).mean() / scaling
        model.backward(loss)
        optimizer.step()

        # store checkpoint and pass link to model_pool