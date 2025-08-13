import ray, torch, tree, random, math
from typing import Optional, Dict
from dataclasses import replace
from unstable.learners.base import BaseLearner
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt
from unstable.reward_transformations.transformation_sampling import NormalizeRewardsByEnv

def compute_gae(rewards, values, gamma=1.0, gae_lambda=1.0): # Compute gae (for policy learning) and return (for critic learning)
    assert len(rewards) == len(values)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 0  # Assume a complete episode
            nextvalues = 0  # Does not matter
        else:
            nextnonterminal = 1.0
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages

@ray.remote
class PPOLearner(BaseLearner):
    def initialize_algorithm(self, ppo_cfg: dict, infer_mini_batch_size: int, normalize_adv: bool=False, max_generation_len: Optional[int]=None, max_train_len: Optional[int]=None, initial_lora_path: Optional[str]=None):
        # PPO specific parameters
        self.gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
        self.gamma = ppo_cfg.get("gamma", 0.99)
        self.clip_coef = ppo_cfg.get("clip_coef", 0.2)
        self.ent_coef = ppo_cfg.get("ent_coef", 0.01)
        self.vf_coef = ppo_cfg.get("vf_coef", 0.5)
        self.update_epochs = ppo_cfg.get("update_epochs", 4)
        self.max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
        critic_learning_rate = ppo_cfg.get("critic_learning_rate", 5e-5)
        self.infer_mini_batch_size = infer_mini_batch_size
        self.normalize_adv = normalize_adv
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len

        # build the critic
        self.critic, _ = build_peft_model(self.model_name, self.device, self.lora_cfg, initial_lora_path, critic_model=True)
        if not self.use_trainer_cache:      self.policy_model.config.use_cache = False
        if self.gradient_checkpointing:     self.policy_model.gradient_checkpointing_enable() # gradient checkpointing
        if self.activation_checkpointing:   enable_full_activation_ckpt(self.policy_model)       # activation checkpointing. Affords most of the vRAM savings
        self.critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.critic.parameters()), lr=critic_learning_rate,)

    def _prepare_batch(self, steps):
        # filter steps with missing/invalid logp
        clean_steps = []
        dropped = 0
        for s in steps:
            lp = getattr(s, "logp", None)
            if lp is None or (isinstance(lp, float) and (math.isnan(lp) or math.isinf(lp))):
                dropped += 1
                continue
            clean_steps.append(s)
        if dropped > 0:
            self.logger.warning(f"Dropped {dropped}/{len(steps)} steps due to invalid old_logps")
        if len(clean_steps) == 0:
            raise RuntimeError("All steps in minibatch have invalid/missing old_logps")
        obs, acts, advs, rets, old_logps = zip(*[(s.obs, s.act, s.reward, s.step_info.get("return", torch.nan), s.logp) for s in clean_steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        rets = torch.tensor(rets, dtype=torch.float32, device=self.device)
        old_logps = torch.tensor(old_logps, dtype=torch.float32, device=self.device)

        combined = [o + a for o, a in zip(obs, acts)]
        lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
        avg_len = sum(lengths) / len(lengths)
        pct_truncated = (sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0)
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device)
        state_enc = self.tokenizer(obs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device)
        return enc, state_enc, advs, rets, old_logps, obs, avg_len, pct_truncated

    def _mini_batch_update_step(self, steps, scaling: float = 1.0):
        enc, state_enc, advs, rets, old_logps, obs, avg_len, pct_truncated = self._prepare_batch(steps=steps)
        
        # --- Policy and Entropy Calculation ---
        out = self.policy_model(**enc)
        logp_dist = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp_dist[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        
        # Build mask from attention_mask to exclude padding, then remove prompt tokens
        mask = enc.attention_mask.bool()
        for i, o in enumerate(obs):
            prompt_len = len(self.tokenizer(o, add_special_tokens=False)["input_ids"])
            mask[i, :prompt_len] = False
        mask = mask[:, 1:]

        # Normalize by actual number of generated (non-pad, non-prompt) tokens
        denom = mask.sum(1).clamp_min(1)
        if mask.sum() == 0:
            self.logger.warning("Mask has zero valid tokens; check tokenization and max_train_len")
        new_logps = (tok_logp * mask).sum(1) / denom

        # Entropy calculation aligned to token positions used in tok_logp/mask
        token_logp = logp_dist[:, :-1, :]
        prob_dist = torch.exp(token_logp)
        token_entropy = -(prob_dist * token_logp).sum(-1)  # [B, T-1]
        entropy = (token_entropy * mask).sum() / mask.sum()

        # --- Value Function Calculation ---
        value_pred = self.critic(**state_enc)[:, 0]
        v_loss = 0.5 * ((value_pred - rets) ** 2).mean()

        # --- PPO Policy Loss Calculation ---
        logratio = new_logps - old_logps
        ratio = torch.exp(logratio)
        pg_loss1 = -advs * ratio
        pg_loss2 = -advs * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # --- Total Loss ---
        loss = (pg_loss - self.ent_coef * entropy + self.vf_coef * v_loss) / scaling
        loss.backward()
        torch.cuda.empty_cache()

        clip_high = 1 + self.clip_coef
        clip_low = 1 - self.clip_coef
        clip_frac = ((ratio > clip_high) | (ratio < clip_low)).float().mean().item()
        return {
            "policy_loss": pg_loss.item(), "value_loss": v_loss.item(), "entropy": entropy.item(),
            "logp_mean": new_logps.mean().item(), "value_mae": (value_pred-rets).abs().mean().item(),
            "logp_std": new_logps.std().item(), "num_steps": len(clean_steps), "avg_train_len": avg_len, "pct_truncated": pct_truncated,
            "old_logp_mean": old_logps.mean().item(), "ratio_mean": ratio.mean().item(), "ratio_std": ratio.std().item(), "clip_frac": clip_frac,
            "denom_mean": denom.float().mean().item(),
        }


    def _update(self, batch):
        all_samples = tree.flatten(batch)
        num_samples = len(all_samples)

        # --- GAE and Return Calculation ---
        all_values = []
        for i in range(0, len(all_samples), self.infer_mini_batch_size):
            batch_steps = all_samples[i : i + self.infer_mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                _, state_enc, _, _, _, _, _, _ = self._prepare_batch(batch_steps)
                with torch.no_grad():
                    target_values = self.critic(**state_enc)[:, 0]
                    all_values.append(target_values)
        all_values = torch.cat(all_values).float().cpu()
        ep_values = torch.split(all_values, [len(ep) for ep in batch])
        ep_rewards = [torch.tensor([step.reward for step in ep]) for ep in batch]
        
        ep_advantages = []
        ep_returns = []
        for rewards, values in zip(ep_rewards, ep_values):
            adv = compute_gae(rewards, values, self.gamma, self.gae_lambda)
            ep_advantages.append(adv)
            ep_returns.append(adv + values)

        # --- Prepare Training Batch ---
        train_batch = []
        for i, ep in enumerate(batch):
            for j, step in enumerate(ep):
                step = replace(step, reward=ep_advantages[i][j].item())
                step = replace(step, step_info={**step.step_info, "return": ep_returns[i][j].item(), "advantage": ep_advantages[i][j].item()})
                train_batch.append(step)
        
        if self.normalize_adv: train_batch = NormalizeRewardsByEnv(True)(train_batch)

        # --- PPO Multi-Epoch Update ---
        metrics_acc: Dict[str, float] = {}
        num_mini_batches = 0

        for epoch in range(self.update_epochs):
            random.shuffle(train_batch)
            for i in range(0, len(train_batch), self.mini_batch_size):
                self.policy_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                sub_batch = train_batch[i : i + self.mini_batch_size]
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    update_metrics = self._mini_batch_update_step(sub_batch)
                
                for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                num_mini_batches += 1

                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.policy_optimizer.step()
                self.critic_optimizer.step()

        self._step += 1
        self.logger.info(f"PPO update complete. Ran {self.update_epochs} epochs with {num_mini_batches} total mini-batches.")

        # --- Logging ---
        log = {f"{k}": v / num_mini_batches for k, v in metrics_acc.items()}
        grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.policy_model.parameters() if p.grad is not None) ** 0.5)
        critic_grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.critic.parameters() if p.grad is not None)** 0.5)
        log.update({
            "step": self._step, "samples_seen": self._samples_seen, "lr": self.policy_optimizer.param_groups[0]["lr"], 
            "grad_norm": grad_norm, "critic_lr": self.critic_optimizer.param_groups[0]["lr"], "critic_grad_norm": critic_grad_norm,
        })
        return log




