import ray, torch, tree, random, os
from typing import Optional
from dataclasses import replace
import torch.nn as nn
from unstable.learners.base import BaseLearner
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt
from unstable.reward_transformations.transformation_sampling import NormalizeRewardsByEnv

class SharedBackbonePPOModel(nn.Module):
    """共享backbone的PPO模型，包含策略头和价值头"""
    
    def __init__(self, base_model, tokenizer, device, value_head_hidden_size=512):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        
        # 获取模型的隐藏层大小
        if hasattr(base_model.config, 'hidden_size'):
            hidden_size = base_model.config.hidden_size
        elif hasattr(base_model.config, 'd_model'):
            hidden_size = base_model.config.d_model
        else:
            hidden_size = 4096  # 默认值
        
        # 价值头：简单的线性层
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, value_head_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(value_head_hidden_size, 1)
        ).to(device)
        
        # 初始化价值头权重
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, input_ids, attention_mask=None, return_value_only=False, return_policy_only=False):
        """前向传播
        
        Args:
            return_value_only: 只返回价值估计
            return_policy_only: 只返回策略logits
        """
        # 共享backbone前向传播
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        if return_policy_only:
            return outputs
        
        # 获取最后一层隐藏状态用于价值估计
        last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        if return_value_only:
            # 只计算价值，使用第一个token的表示
            first_token_hidden = last_hidden_states[:, 0, :]  # [batch_size, hidden_size]
            values = self.value_head(first_token_hidden).squeeze(-1)  # [batch_size]
            return values
        
        # 同时返回策略和价值
        first_token_hidden = last_hidden_states[:, 0, :]
        values = self.value_head(first_token_hidden).squeeze(-1)
        
        return outputs, values
    
    def get_policy_logits(self, input_ids, attention_mask=None):
        """获取策略logits"""
        outputs = self.forward(input_ids, attention_mask, return_policy_only=True)
        return outputs.logits
    
    def get_values(self, input_ids, attention_mask=None):
        """获取价值估计"""
        return self.forward(input_ids, attention_mask, return_value_only=True)
    
    def save_pretrained(self, save_directory, **kwargs):
        """保存模型"""
        # 保存base model
        self.base_model.save_pretrained(save_directory, **kwargs)
        
        # 保存价值头
        value_head_path = f"{save_directory}/value_head.pt"
        torch.save(self.value_head.state_dict(), value_head_path)
    
    def load_value_head(self, save_directory):
        """加载价值头"""
        value_head_path = f"{save_directory}/value_head.pt"
        if os.path.exists(value_head_path):
            self.value_head.load_state_dict(torch.load(value_head_path, map_location=self.device))

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
                           value_loss_coef: float=0.5, kl_target: float=0.01, kl_coef: float=0.2,
                           # 新增内存优化参数
                           max_seq_len: Optional[int]=4096, memory_efficient_mode: bool=True,
                           gradient_accumulation_steps: int=1):
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
        
        # 内存优化参数
        self.max_seq_len = max_seq_len
        self.memory_efficient_mode = memory_efficient_mode
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # 构建共享backbone模型
        base_model, self.tokenizer = build_peft_model(self.model_name, self.device, self.lora_cfg, initial_lora_path)
        
        # 创建共享backbone + 双头模型
        self.shared_model = SharedBackbonePPOModel(
            base_model=base_model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        # 配置模型设置
        if not self.use_trainer_cache:
            self.shared_model.base_model.config.use_cache = False
        if self.gradient_checkpointing:
            self.shared_model.base_model.gradient_checkpointing_enable()
        if self.activation_checkpointing:
            enable_full_activation_ckpt(self.shared_model.base_model)
        
        # 设置优化器 - 分别为策略和价值头设置不同学习率
        policy_params = list(self.shared_model.base_model.parameters())
        value_params = list(self.shared_model.value_head.parameters())
        
        self.policy_optimizer = torch.optim.AdamW(policy_params, lr=learning_rate)
        self.critic_optimizer = torch.optim.AdamW(value_params, lr=critic_learning_rate)
        
        # 保持向后兼容性
        self.policy_model = self.shared_model  # 为了兼容现有代码
        self.critic = self.shared_model  # 为了兼容现有代码

    def _prepare_batch(self, steps):
        obs, acts, advs, rets, old_logps = zip(*[(s.obs, s.act, s.reward, s.step_info.get("return", torch.nan), s.step_info.get("old_logp", 0.0)) for s in steps])
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        rets = torch.tensor(rets, dtype=torch.float32, device=self.device)
        old_logps = torch.tensor(old_logps, dtype=torch.float32, device=self.device)
        combined = [o + a for o, a in zip(obs, acts)]
        
        # 内存优化：动态调整序列长度
        if self.memory_efficient_mode:
            lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
            # 使用95%分位数作为动态最大长度，避免极长序列
            dynamic_max_len = min(int(sorted(lengths)[int(len(lengths) * 0.95)]), 
                                self.max_train_len or self.max_seq_len)
            avg_len = sum(lengths) / len(lengths)
            pct_truncated = sum(l > dynamic_max_len for l in lengths) / len(lengths)
        else:
            lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
            dynamic_max_len = self.max_train_len
            avg_len = sum(lengths) / len(lengths)
            pct_truncated = (sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0)
        
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True, max_length=dynamic_max_len).to(self.device)
        state_enc = self.tokenizer(obs, return_tensors="pt", padding=True, truncation=True, max_length=dynamic_max_len).to(self.device)
        return enc, state_enc, advs, rets, old_logps, obs, avg_len, pct_truncated

    def _mini_batch_update_step(self, steps, scaling: float = 1.0):
        enc, state_enc, advs, rets, old_logps, obs, avg_len, pct_truncated = self._prepare_batch(steps=steps)
        device = enc.input_ids.device

        # 前向传播 - 策略部分
        policy_outputs = self.shared_model.get_policy_logits(enc.input_ids, enc.attention_mask)
        logp_full = torch.nn.functional.log_softmax(policy_outputs, dim=-1)          # [B, T, V]

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

        # 值函数 - 使用独立的前向传播避免梯度干扰
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            value_pred = self.shared_model.get_values(state_enc.input_ids, state_enc.attention_mask)
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

        # 内存优化：分块处理值函数和旧策略概率计算
        if self.memory_efficient_mode:
            return self._memory_efficient_update(batch, all_samples, num_samples)
        
        # 原始方法：构建值函数目标和存储旧的log概率
        all_values = []
        all_old_logps = []
        
        for i in range(0, len(all_samples), self.infer_mini_batch_size):
            batch_steps = all_samples[i : i + self.infer_mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                enc, state_enc, _, _, _, obs, _, _ = self._prepare_batch(batch_steps)
                with torch.no_grad():
                    # 值函数估计
                    target_values = self.shared_model.get_values(state_enc.input_ids, state_enc.attention_mask)
                    all_values.append(target_values)
                    
                    # 存储旧策略的log概率
                    policy_logits = self.shared_model.get_policy_logits(enc.input_ids, enc.attention_mask)
                    logp = torch.nn.functional.log_softmax(policy_logits, dim=-1)
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
            
            # 梯度裁剪和优化器步骤 - 分别处理策略和价值参数
            torch.nn.utils.clip_grad_norm_(self.shared_model.base_model.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.shared_model.value_head.parameters(), self.grad_clip)
            
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
        
        grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.shared_model.base_model.parameters() if p.grad is not None) ** 0.5)
        critic_grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.shared_model.value_head.parameters() if p.grad is not None) ** 0.5)
        
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

    def _memory_efficient_update(self, batch, all_samples, num_samples):
        """内存优化的更新方法：分块处理，避免大张量累积"""
        # 分块计算值函数和旧策略概率，避免内存累积
        chunk_size = min(self.infer_mini_batch_size * 2, 64)  # 更小的块大小
        
        # 存储每个episode的处理结果
        ep_advantages = []
        ep_returns = []
        ep_old_logps = []
        
        # 按episode分组处理
        ep_start_idx = 0
        for ep in batch:
            ep_end_idx = ep_start_idx + len(ep)
            ep_samples = all_samples[ep_start_idx:ep_end_idx]
            
            # 分块处理当前episode
            ep_values = []
            ep_logps = []
            
            for i in range(0, len(ep_samples), chunk_size):
                chunk_samples = ep_samples[i:i + chunk_size]
                
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    enc, state_enc, _, _, _, obs, _, _ = self._prepare_batch(chunk_samples)
                    
                    with torch.no_grad():
                        # 值函数估计
                        chunk_values = self.shared_model.get_values(state_enc.input_ids, state_enc.attention_mask).float().cpu()
                        ep_values.append(chunk_values)
                        
                        # 旧策略概率
                        policy_logits = self.shared_model.get_policy_logits(enc.input_ids, enc.attention_mask)
                        logp = torch.nn.functional.log_softmax(policy_logits, dim=-1)
                        tgt_ids = enc.input_ids[:, 1:]
                        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
                        
                        # 构建mask
                        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device)
                        for j, o in enumerate(obs):
                            prompt_len = len(self.tokenizer(o, add_special_tokens=False)["input_ids"])
                            mask[j, :prompt_len] = False
                        mask = mask[:, 1:]
                        
                        # 计算序列级别的log概率
                        valid_tokens = mask.sum(dim=1).clamp_min(1)
                        seq_logp = (tok_logp * mask).sum(dim=1) / valid_tokens
                        ep_logps.append(seq_logp.float().cpu())
                
                # 立即清理GPU缓存
                torch.cuda.empty_cache()
            
            # 合并当前episode的结果
            ep_values_tensor = torch.cat(ep_values)
            ep_logps_tensor = torch.cat(ep_logps)
            ep_rewards = torch.tensor([step.reward for step in ep])
            
            # 计算GAE
            adv, ret = compute_gae(ep_rewards, ep_values_tensor, last_value=0.0, done=True)
            ep_advantages.append(adv)
            ep_returns.append(ret)
            ep_old_logps.append(ep_logps_tensor)
            
            ep_start_idx = ep_end_idx
        
        # 构建训练批次
        train_batch = []
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
            from unstable.reward_transformations.transformation_sampling import NormalizeRewardsByEnv
            train_batch = NormalizeRewardsByEnv(True)(train_batch)
        
        # 内存优化的PPO更新
        return self._memory_efficient_ppo_update(train_batch, num_samples)
    
    def _memory_efficient_ppo_update(self, train_batch, num_samples):
        """内存优化的PPO多轮更新"""
        metrics_acc = {}
        train_batch = random.sample(train_batch, self.batch_size)
        
        # 调整mini_batch_size以适应内存限制
        effective_mini_batch_size = max(self.mini_batch_size // 2, 4) if self.memory_efficient_mode else self.mini_batch_size
        num_steps = self.batch_size // effective_mini_batch_size
        
        self.logger.info(f"Memory-efficient mode: {num_samples} samples, {self.ppo_epochs} epochs, {num_steps} steps/epoch, mini_batch_size={effective_mini_batch_size}")
        
        for epoch in range(self.ppo_epochs):
            # 梯度累积模式
            self.policy_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            
            random.shuffle(train_batch)
            epoch_metrics = {}
            
            for i in range(num_steps):
                sub = train_batch[i * effective_mini_batch_size : (i + 1) * effective_mini_batch_size]
                
                # 使用梯度累积减少内存压力
                accumulation_steps = max(1, self.gradient_accumulation_steps)
                sub_chunks = [sub[j:j + len(sub)//accumulation_steps] 
                             for j in range(0, len(sub), len(sub)//accumulation_steps)]
                
                for chunk_idx, chunk in enumerate(sub_chunks):
                    if not chunk:
                        continue
                        
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        scaling_factor = num_steps * self.ppo_epochs * len(sub_chunks)
                        update_metrics = self._mini_batch_update_step(chunk, scaling=scaling_factor)
                    
                    # 累积指标
                    for k, v in update_metrics.items():
                        epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v / len(sub_chunks)
                        metrics_acc[k] = metrics_acc.get(k, 0.0) + v / len(sub_chunks)
                    
                    # 每个chunk后清理缓存
                    torch.cuda.empty_cache()
            
            # 梯度裁剪和优化 - 分别处理策略和价值参数
            torch.nn.utils.clip_grad_norm_(self.shared_model.base_model.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.shared_model.value_head.parameters(), self.grad_clip)
            
            try:
                self.policy_optimizer.step()
                self.critic_optimizer.step()
            except Exception as exc:
                self.logger.exception(f"optimizer.step failed on step {self._step}, epoch {epoch}: {exc}")
                raise
            
            # KL散度早停检查
            avg_kl = epoch_metrics.get("kl_div", 0.0) / num_steps
            if avg_kl > 1.5 * self.kl_target:
                self.logger.info(f"Early stopping at epoch {epoch} due to high KL: {avg_kl:.4f}")
                break
            
            self.logger.info(f"Epoch {epoch} avg metrics: {{{k}: {v/num_steps:.4f} for k, v in epoch_metrics.items()}}")
        
        self._step += 1
        
        # 计算最终指标
        total_updates = (epoch + 1) * num_steps
        log = {f"{k}": v / total_updates for k, v in metrics_acc.items()}
        
        # 添加梯度范数
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.shared_model.base_model.parameters() if p.grad is not None) ** 0.5
        critic_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.shared_model.value_head.parameters() if p.grad is not None) ** 0.5
        
        log.update({
            "step": self._step,
            "samples_seen": self._samples_seen,
            "lr": self.policy_optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            "critic_grad_norm": critic_grad_norm,
            "ppo_epochs_completed": epoch + 1,
            "effective_mini_batch_size": effective_mini_batch_size,
        })
        
        return log