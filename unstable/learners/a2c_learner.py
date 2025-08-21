import ray, torch, tree, random
from typing import Optional
from dataclasses import replace
from unstable.learners.base import BaseLearner
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt
from unstable.reward_transformations.transformation_sampling import NormalizeRewardsByEnv

def compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95, last_value=0.0, done=True):
    """
    计算广义优势估计 (Generalized Advantage Estimation)
    
    参数:
        rewards: 奖励序列 [T]
        values: 值函数估计 [T]
        gamma: 折扣因子
        gae_lambda: GAE lambda参数
        last_value: 序列结束后的估计值（用于bootstrap）
        done: 是否为episode终止状态
        
    返回:
        advantages: 优势函数估计 [T]
        returns: 回报估计 [T]
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 0.0 if done else 1.0
            nextvalues = last_value
        else:
            nextnonterminal = 1.0
            nextvalues = values[t + 1]
            
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
    returns = advantages + values
    return advantages, returns

@ray.remote
class A2CLearner(BaseLearner):
    def initialize_algorithm(self, infer_mini_batch_size: int, critic_learning_rate: float, normalize_adv: bool=False, max_generation_len: Optional[int]=None, max_train_len: Optional[int]=None, initial_lora_path: Optional[str]=None):
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
        obs, acts, advs, rets = zip(*[(s.obs, s.act, s.reward, s.step_info.get("return", torch.nan)) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        rets = torch.tensor(rets, dtype=torch.float32, device=self.device)
        combined = [o + a for o, a in zip(obs, acts)]
        lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
        avg_len = sum(lengths) / len(lengths)
        pct_truncated = (sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0)
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device)  # Tokenize with truncation
        state_enc = self.tokenizer(obs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_train_len).to(self.device)  # Tokenize with truncation
        return enc, state_enc, advs, rets, obs, avg_len, pct_truncated

    # def _mini_batch_update_step(self, steps, scaling: float = 1.0):
    #     enc, state_enc, advs, rets, obs, avg_len, pct_truncated = self._prepare_batch(steps=steps)
    #     # Learn policy
    #     out = self.policy_model(**enc)
    #     logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
    #     tgt_ids = enc.input_ids[:, 1:]
    #     tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
    #     mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device)  # build prompt mask
    #     for i, o in enumerate(obs): mask[i, : len(self.tokenizer(o, add_special_tokens=False)["input_ids"])] = False
    #     mask = mask[:, 1:]
    #     seq_logp = (tok_logp * mask).sum(1) / self.max_generation_len
    #     loss = -(advs * seq_logp).mean() / scaling
    #     loss.backward()
    #     torch.cuda.empty_cache()

    #     # Learn value
    #     value_pred = self.critic(**state_enc)[:, 0]
    #     value_loss = 0.5 * ((value_pred - rets) ** 2).mean()
    #     value_loss.backward()
    #     torch.cuda.empty_cache()

    #     return {
    #         "policy_loss": loss.item(), "value_loss": value_loss.item(), "logp_mean": seq_logp.mean().item(), "value_mae": (value_pred-rets).abs().mean().item(), "value_dir_acc": ((value_pred > 0) == (rets > 0)).float().mean().item(),
    #         "logp_std": seq_logp.std().item(), "num_steps": len(steps), "avg_train_len": avg_len, "pct_truncated": pct_truncated,
    #     }
    def _mini_batch_update_step(self, steps, scaling: float = 1.0):
        enc, state_enc, advs, rets, obs, avg_len, pct_truncated = self._prepare_batch(steps=steps)
        device = enc.input_ids.device

        # 1) 前向 + log_softmax
        out = self.policy_model(**enc)
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)          # [B, T, V]
        tgt_ids = enc.input_ids[:, 1:]                                      # [B, T-1]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        # 2) 精确构造 label_mask：非pad 且 非prompt 部分
        attn = enc.attention_mask.bool()                                     # [B, T]
        # prompt 长度：用 tokenizer 逐条算
        prompt_lens = torch.tensor(
            [len(self.tokenizer(o, add_special_tokens=False)["input_ids"]) for o in obs],
            device=device
        )
        token_idx = torch.arange(attn.size(1), device=device).unsqueeze(0)   # [1, T]
        prompt_mask = token_idx < prompt_lens.unsqueeze(1)                   # True=prompt 区
        label_mask = (attn & (~prompt_mask))[:, 1:]                          # 目标对齐到 tgt_ids / tok_logp

        # 3) 避免 0 * (-inf) → 先把无效位替换为 0，再求和
        tok_logp = tok_logp.masked_fill(~label_mask, 0.0)                    # 替换而非乘法
        valid_tokens = label_mask.sum(dim=1).clamp_min(1)                    # 每条样本有效 token 数
        seq_logp = tok_logp.sum(dim=1) / valid_tokens                        # per-sample 平均 logp

        # 4) 数值稳健：把 NaN/Inf 清理掉（若仍出现）
        seq_logp = torch.nan_to_num(seq_logp, nan=0.0, posinf=30.0, neginf=-30.0)
        advs = torch.nan_to_num(advs, nan=0.0, posinf=1e4, neginf=-1e4)

        # （可选）优势标准化，进一步稳住策略损失
        if getattr(self, "normalize_adv", False):
            advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)

        # 5) 策略 loss
        loss = -(advs * seq_logp).mean() / scaling
        loss.backward()
        torch.cuda.empty_cache()

        # 6) 值函数
        value_pred = self.critic(**state_enc)[:, 0]
        value_loss = 0.5 * ((value_pred - rets) ** 2).mean()
        value_loss.backward()
        torch.cuda.empty_cache()

        # 7) 记录指标（注意 std 用 unbiased=False 避免 batch=1 时 NaN）
        logp_mean = seq_logp.mean().item()
        logp_std = seq_logp.std(unbiased=False).item()
        value_mae = (value_pred - rets).abs().mean().item()
        value_dir_acc = ((value_pred > 0) == (rets > 0)).float().mean().item()

        return {
            "policy_loss": loss.item(),
            "value_loss": value_loss.item(),
            "logp_mean": logp_mean,
            "value_mae": value_mae,
            "value_dir_acc": value_dir_acc,
            "logp_std": logp_std,
            "num_steps": len(steps),
            "avg_train_len": avg_len,
            "pct_truncated": pct_truncated,
        }


    def _update(self, batch):
        all_samples = tree.flatten(batch)
        num_samples = len(all_samples)

        # construct value targets
        all_values = []
        for i in range(0, len(all_samples), self.infer_mini_batch_size):
            batch_steps = all_samples[i : i + self.infer_mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                _, state_enc, _, _, _, _, _ = self._prepare_batch(batch_steps)
                with torch.no_grad():
                    # take the first token's output
                    target_values = self.critic(**state_enc)[:, 0]
                    all_values.append(target_values)
        all_values = torch.cat(all_values).float().cpu()
        ep_values = torch.split(all_values, [len(ep) for ep in batch])
        ep_rewards = [torch.tensor([step.reward for step in ep]) for ep in batch]
        ep_advantages = []
        ep_returns = []
        for rewards, values in zip(ep_rewards, ep_values):
            # 假设每个episode都是完整的，如果有截断episode需要传入done=False
            adv, ret = compute_gae(rewards, values, last_value=0.0, done=True)
            ep_advantages.append(adv)
            ep_returns.append(ret)

        train_batch = []
        for i, ep in enumerate(batch):
            for j, step in enumerate(ep):
                step = replace(step, reward=ep_advantages[i][j].item())
                step = replace(step, step_info={**step.step_info, "return": ep_returns[i][j].item(), "advantage": ep_advantages[i][j].item()})
                train_batch.append(step)
        assert len(train_batch) >= self.batch_size

        self.policy_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

        metrics_acc: Dict[str, float] = {}

        train_batch = random.sample(train_batch, self.batch_size)
        num_steps = self.batch_size // self.mini_batch_size
        self.logger.info(f"Got {num_samples} many samples. Running for {num_steps} steps (i.e. mini batch size: {self.mini_batch_size})")

        if self.normalize_adv: train_batch = NormalizeRewardsByEnv(True)(train_batch)
        for i in range(num_steps):
            sub = train_batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                update_metrics = self._mini_batch_update_step(sub, scaling=num_steps)
            for k, v in update_metrics.items(): metrics_acc[k] = metrics_acc.get(k, 0.0) + v
            self.logger.info(f"Mini-step metrics: {update_metrics}")
        self.logger.info(f"Step metrics: {metrics_acc}")

        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

        try:
            self.policy_optimizer.step()
            self.critic_optimizer.step()
        except Exception as exc:
            self.logger.exception(f"optimizer.step crashed on step {self._step} -\n\n{exc}\n\n")
            raise
        self._step += 1

        log = {f"{k}": v / num_steps for k, v in metrics_acc.items()}
        grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.policy_model.parameters() if p.grad is not None) ** 0.5)
        critic_grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.critic.parameters() if p.grad is not None)** 0.5)
        log.update({
            "step": self._step, "samples_seen": self._samples_seen, "lr": self.policy_optimizer.param_groups[0]["lr"], 
            "grad_norm": grad_norm, "critic_lr": self.critic_optimizer.param_groups[0]["lr"], "critic_grad_norm": critic_grad_norm,
        })
        return log








