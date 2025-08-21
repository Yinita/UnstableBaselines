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
class PPOLearner(BaseLearner):
    def initialize_algorithm(self, infer_mini_batch_size: int, critic_learning_rate: float, 
                           normalize_adv: bool=False, max_generation_len: Optional[int]=None, 
                           max_train_len: Optional[int]=None, initial_lora_path: Optional[str]=None,
                           clip_ratio: float=0.2, ppo_epochs: int=4, entropy_coef: float=0.01,
                           value_loss_coef: float=0.5, kl_target: float=0.01, kl_coef: float=0.2):
        """
        初始化PPO算法参数
        
        参数:
            clip_ratio: PPO裁剪比率
            ppo_epochs: 每个batch的PPO更新轮数
            entropy_coef: 熵正则化系数
            value_loss_coef: 值函数损失权重
            kl_target: KL散度目标值
            kl_coef: KL散度惩罚系数
        """
        self.infer_mini_batch_size = infer_mini_batch_size
        self.normalize_adv = normalize_adv
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len
        
        # PPO特有参数
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.kl_target = kl_target
        self.kl_coef = kl_coef

        # build the critic
        self.critic, _ = build_peft_model(self.model_name, self.device, self.lora_cfg, initial_lora_path, critic_model=True)
        if not self.use_trainer_cache:      self.critic.config.use_cache = False
        if self.gradient_checkpointing:     self.critic.gradient_checkpointing_enable()
        if self.activation_checkpointing:   enable_full_activation_ckpt(self.critic)
        self.critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.critic.parameters()), lr=critic_learning_rate)

    def _prepare_batch(self, steps):
        obs, acts, advs, rets, old_logps = zip(*[(s.obs, s.act, s.reward, s.step_info.get("return", torch.nan), s.step_info.get("old_logp", 0.0)) for s in steps])
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
        device = enc.input_ids.device

        # 前向
        out = self.policy_model(**enc)
        logp_full = torch.nn.functional.log_softmax(out.logits, dim=-1)          # [B, T, V]

        # 目标 token 对齐：去掉最后一位（因右移）
        tgt_ids  = enc.input_ids[:, 1:]                                          # [B, T-1]
        tok_logp = logp_full[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        # ---- 构造 label_mask：非pad 且 非prompt ----
        attn = enc.attention_mask.bool()                                         # [B, T]
        prompt_lens = torch.tensor(
            [len(self.tokenizer(o, add_special_tokens=False)["input_ids"]) for o in obs],
            device=device
        )
        token_idx = torch.arange(attn.size(1), device=device).unsqueeze(0)       # [1, T]
        prompt_mask = token_idx < prompt_lens.unsqueeze(1)                       # True=prompt
        label_mask = (attn & (~prompt_mask))[:, 1:]                              # [B, T-1]

        # ---- 避免 0 * (-inf)：先把无效位替换为 0，再求和；用每样本有效 token 数作分母 ----
        tok_logp = tok_logp.masked_fill(~label_mask, 0.0)
        valid_tokens = label_mask.sum(dim=1).clamp_min(1)
        seq_logp = tok_logp.sum(dim=1) / valid_tokens                            # [B]

        # （可选）优势标准化
        if getattr(self, "normalize_adv", False):
            advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)

        # ---- 熵：逐位置熵，再按相同 mask 聚合 ----
        probs = torch.nn.functional.softmax(out.logits, dim=-1)                  # [B, T, V]
        token_entropy = -(probs * logp_full).sum(-1)                             # [B, T]
        token_entropy = token_entropy[:, :-1]                                    # [B, T-1]
        token_entropy = token_entropy.masked_fill(~label_mask, 0.0)
        seq_entropy = token_entropy.sum(dim=1) / valid_tokens                    # [B]

        # 数值保底
        seq_logp   = torch.nan_to_num(seq_logp,   nan=0.0, posinf=30.0, neginf=-30.0)
        old_logps  = torch.nan_to_num(old_logps,  nan=0.0, posinf=30.0, neginf=-30.0)
        seq_entropy= torch.nan_to_num(seq_entropy,nan=0.0, posinf=30.0, neginf=0.0)

        # ---- PPO ratio，防溢出 ----
        diff  = (seq_logp - old_logps).clamp(-20.0, 20.0)
        ratio = torch.exp(diff)

        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advs
        policy_loss = -torch.min(surr1, surr2).mean()

        # 熵正则（加号：鼓励熵）
        entropy_loss = -seq_entropy.mean()

        # KL 惩罚（注意口径一致）
        kl_div = (old_logps - seq_logp).mean()
        kl_penalty = self.kl_coef * torch.clamp(kl_div - self.kl_target, min=0.0)

        total_policy_loss = (policy_loss + self.entropy_coef * entropy_loss + kl_penalty) / scaling
        total_policy_loss.backward()
        torch.cuda.empty_cache()

        # 值函数
        value_pred = self.critic(**state_enc)[:, 0]
        value_loss = self.value_loss_coef * 0.5 * ((value_pred - rets) ** 2).mean() / scaling
        value_loss.backward()
        torch.cuda.empty_cache()

        # 统计（避免 batch=1 的 std=NaN）
        clipped_fraction = ((ratio < 1.0 - self.clip_ratio) | (ratio > 1.0 + self.clip_ratio)).float().mean()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "kl_div": kl_div.item(),
            "kl_penalty": kl_penalty.item(),
            "clipped_fraction": clipped_fraction.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std(unbiased=False).item(),
            "logp_mean": seq_logp.mean().item(),
            "logp_std": seq_logp.std(unbiased=False).item(),
            "value_mae": (value_pred - rets).abs().mean().item(),
            "value_dir_acc": ((value_pred > 0) == (rets > 0)).float().mean().item(),
            "num_steps": len(steps),
            "avg_train_len": avg_len,
            "pct_truncated": pct_truncated,
        }


    def _update(self, batch):
        all_samples = tree.flatten(batch)
        num_samples = len(all_samples)

        # 构建值函数目标和存储旧的log概率
        all_values = []
        all_old_logps = []
        
        for i in range(0, len(all_samples), self.infer_mini_batch_size):
            batch_steps = all_samples[i : i + self.infer_mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                enc, state_enc, _, _, _, obs, _, _ = self._prepare_batch(batch_steps)
                with torch.no_grad():
                    # 值函数估计
                    target_values = self.critic(**state_enc)[:, 0]
                    all_values.append(target_values)
                    
                    # 存储旧策略的log概率
                    out = self.policy_model(**enc)
                    logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
                    tgt_ids = enc.input_ids[:, 1:]
                    tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
                    mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device)
                    for j, o in enumerate(obs): 
                        mask[j, : len(self.tokenizer(o, add_special_tokens=False)["input_ids"])] = False
                    mask = mask[:, 1:]
                    seq_logp = (tok_logp * mask).sum(1) / self.max_generation_len
                    all_old_logps.append(seq_logp)
        
        all_values = torch.cat(all_values).float().cpu()
        all_old_logps = torch.cat(all_old_logps).float().cpu()
        
        ep_values = torch.split(all_values, [len(ep) for ep in batch])
        ep_old_logps = torch.split(all_old_logps, [len(ep) for ep in batch])
        ep_rewards = [torch.tensor([step.reward for step in ep]) for ep in batch]
        
        ep_advantages = []
        ep_returns = []
        for rewards, values in zip(ep_rewards, ep_values):
            adv, ret = compute_gae(rewards, values, last_value=0.0, done=True)
            ep_advantages.append(adv)
            ep_returns.append(ret)

        # 构建训练批次，包含旧的log概率
        train_batch = []
        old_logp_idx = 0
        for i, ep in enumerate(batch):
            for j, step in enumerate(ep):
                step = replace(step, reward=ep_advantages[i][j].item())
                step = replace(step, step_info={
                    **step.step_info, 
                    "return": ep_returns[i][j].item(), 
                    "advantage": ep_advantages[i][j].item(),
                    "old_logp": ep_old_logps[i][j].item()
                })
                train_batch.append(step)
        
        assert len(train_batch) >= self.batch_size

        if self.normalize_adv: 
            train_batch = NormalizeRewardsByEnv(True)(train_batch)

        # PPO多轮更新
        metrics_acc = {}
        train_batch = random.sample(train_batch, self.batch_size)
        num_steps = self.batch_size // self.mini_batch_size
        
        self.logger.info(f"Got {num_samples} samples. Running PPO for {self.ppo_epochs} epochs, {num_steps} steps per epoch")

        for epoch in range(self.ppo_epochs):
            self.policy_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            
            # 随机打乱数据
            random.shuffle(train_batch)
            
            epoch_metrics = {}
            for i in range(num_steps):
                sub = train_batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    update_metrics = self._mini_batch_update_step(sub, scaling=num_steps * self.ppo_epochs)
                for k, v in update_metrics.items(): 
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + v
            
            # 梯度裁剪和优化器步骤
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            
            try:
                self.policy_optimizer.step()
                self.critic_optimizer.step()
            except Exception as exc:
                self.logger.exception(f"optimizer.step crashed on step {self._step}, epoch {epoch} - {exc}")
                raise
            
            # 早停检查：如果KL散度太大，提前停止
            avg_kl = epoch_metrics.get("kl_div", 0.0) / num_steps
            if avg_kl > 1.5 * self.kl_target:
                self.logger.info(f"Early stopping at epoch {epoch} due to high KL divergence: {avg_kl:.4f}")
                break
                
            self.logger.info(f"Epoch {epoch} metrics: {{k: v/num_steps for k, v in epoch_metrics.items()}}")

        self._step += 1

        # 计算平均指标
        total_updates = (epoch + 1) * num_steps
        log = {f"{k}": v / total_updates for k, v in metrics_acc.items()}
        
        grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.policy_model.parameters() if p.grad is not None) ** 0.5)
        critic_grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.critic.parameters() if p.grad is not None) ** 0.5)
        
        log.update({
            "step": self._step, 
            "samples_seen": self._samples_seen, 
            "lr": self.policy_optimizer.param_groups[0]["lr"], 
            "grad_norm": grad_norm, 
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"], 
            "critic_grad_norm": critic_grad_norm,
            "ppo_epochs_completed": epoch + 1,
        })
        
        return log