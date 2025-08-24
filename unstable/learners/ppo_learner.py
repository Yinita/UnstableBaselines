import ray, torch, tree, random, os
from typing import Optional
from dataclasses import replace
import torch.nn as nn
from unstable.learners.base import BaseLearner
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt
from unstable.reward_transformations.transformation_sampling import NormalizeRewardsByEnv
import torch.nn.functional as F
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
        
        # 初始化价值头权重 - 使用float32进行正交初始化以避免BFloat16兼容性问题
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                # 保存原始数据类型
                orig_dtype = layer.weight.dtype
                # 临时转换为float32进行正交初始化
                layer.weight.data = layer.weight.data.to(torch.float32)
                nn.init.orthogonal_(layer.weight, gain=1.0)
                # 转换回原始数据类型
                layer.weight.data = layer.weight.data.to(orig_dtype)
                # 偏置初始化
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, input_ids, attention_mask=None, return_value_only=False, return_policy_only=False):
        """前向传播
        
        Args:
            return_value_only: 只返回价值估计
            return_policy_only: 只返回策略logits
        """
        # 共享backbone前向传播
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        if return_policy_only:
            return outputs
        
        # 获取最后一层隐藏状态用于价值估计
        last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        if return_value_only:
            # 只计算价值，使用masked-mean pooling 而非首token
            if attention_mask is None:
                # 若无mask，等权平均
                mask = torch.ones(last_hidden_states.size()[:2], device=last_hidden_states.device, dtype=last_hidden_states.dtype)
            else:
                mask = attention_mask.to(last_hidden_states.device).to(last_hidden_states.dtype)
            masked_sum = (last_hidden_states * mask.unsqueeze(-1)).sum(dim=1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled_hidden = masked_sum / denom  # [batch_size, hidden_size]
            # 确保与价值头在同一设备
            value_head_device = next(self.value_head.parameters()).device
            pooled_hidden = pooled_hidden.to(value_head_device)
            values = self.value_head(pooled_hidden).squeeze(-1)  # [batch_size]
            return values
        
        # 同时返回策略和价值
        # 确保输入到value_head的张量在正确的设备上
        value_head_device = next(self.value_head.parameters()).device
        if attention_mask is None:
            mask = torch.ones(last_hidden_states.size()[:2], device=last_hidden_states.device, dtype=last_hidden_states.dtype)
        else:
            mask = attention_mask.to(last_hidden_states.device).to(last_hidden_states.dtype)
        masked_sum = (last_hidden_states * mask.unsqueeze(-1)).sum(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_hidden = masked_sum / denom
        pooled_hidden = pooled_hidden.to(value_head_device)
        values = self.value_head(pooled_hidden).squeeze(-1)
        
        return outputs, values
    
    def get_policy_logits(self, input_ids, attention_mask=None):
        """获取策略logits"""
        outputs = self.forward(input_ids, attention_mask, return_policy_only=True)
        if hasattr(outputs, 'logits'):
            return outputs.logits
        if isinstance(outputs, (tuple, list)):
            return outputs[0]  # 取 logits
        return outputs  # 假定是 logits tensor
    
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
    # 强制使用float32确保类型一致性
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    
    # 确保values和rewards都是float32
    last_value = values.new_tensor(last_value)
    # 确保所有张量在同一设备上
    target_device = last_value.device
    rewards = rewards.to(target_device, dtype=torch.float32)
    values = values.to(target_device, dtype=torch.float32)
    lastgaelam = 0.0
    
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
    def initialize_algorithm(self, infer_mini_batch_size: int=32, learning_rate: float=5e-5, critic_learning_rate: float=1e-4, normalize_adv: bool=True, max_generation_len: Optional[int]=None, 
                           max_train_len: Optional[int]=None, initial_lora_path: Optional[str]=None,
                           clip_ratio: float=0.2, ppo_epochs: int=4, entropy_coef: float=0.002,
                           value_loss_coef: float=0.5, kl_target: float=0.05, kl_coef: float=0.1,
                           # 新增内存优化参数
                           max_seq_len: Optional[int]=4096,
                           gradient_accumulation_steps: int=1, use_fallback_advantages: bool=False):
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_fallback_advantages = use_fallback_advantages

        # 构建共享backbone模型 - 使用device_map="auto"时不指定单一device
        base_model, self.tokenizer = build_peft_model(self.model_name, None, self.lora_cfg, initial_lora_path)
        
        # 创建共享backbone + 双头模型
        self.shared_model = SharedBackbonePPOModel(
            base_model=base_model,
            tokenizer=self.tokenizer,
            device=self.device  # 只用于value_head的设备
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
        obs, acts, advs, rets, old_logps, label_masks = zip(*[
            (
                s.obs,
                s.act,
                s.reward,
                s.step_info.get("return", torch.nan),
                s.step_info.get("old_logp", []),
                s.step_info.get("label_mask", []),   # ⭐ 新增：存下来的 label_mask
            )
            for s in steps
        ])

        advs = torch.tensor(advs, dtype=torch.float32)  # Keep on CPU
        rets = torch.tensor(rets, dtype=torch.float32)  # Keep on CPU

        # 以二者的 max 为该样本长度
        lengths = [
            max(len(lp) if isinstance(lp, list) else 1,
                len(msk) if isinstance(msk, list) else 1)
            for lp, msk in zip(old_logps, label_masks)
        ]
        max_len = max(lengths) if lengths else 1

        padded_logps, padded_masks = [], []
        for logps, mask in zip(old_logps, label_masks):
            if not isinstance(logps, list):
                logps = [float(logps)]
            if not isinstance(mask, list):
                mask = [1] * len(logps)  # 兜底：全部有效

            # 截断或补齐到 max_len
            logps = (logps + [0.0] * max(0, max_len - len(logps)))[:max_len]
            mask  = (mask  + [0]   * max(0, max_len - len(mask))) [:max_len]

            padded_logps.append(logps)
            padded_masks.append(mask)

        old_logps   = torch.tensor(padded_logps, dtype=torch.float32)  # 留在CPU
        label_masks = torch.tensor(padded_masks, dtype=torch.bool)     # 留在CPU


        # --- 拼接 obs+act 作为输入 ---
        combined = [o + a for o, a in zip(obs, acts)]
        pad_len = getattr(self, "_snapshot_pad_len", None)
        if pad_len:
            enc = self.tokenizer(combined, return_tensors="pt",
                                padding="max_length", truncation=True, max_length=pad_len).to(self.device)
            state_enc = self.tokenizer(obs, return_tensors="pt",
                                    padding="max_length", truncation=True, max_length=pad_len).to(self.device)
            # 统计信息仍然按真实长度估算
            raw_lens = [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in combined]
            avg_len = sum(raw_lens) / max(1, len(raw_lens))
            pct_truncated = sum(l > pad_len for l in raw_lens) / max(1, len(raw_lens))
        else:
            lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
            if self.max_train_len is not None:
                dynamic_max_len = self.max_train_len
            else:
                # 95% 分位 + 上限，避免偶发超长
                sorted_l = sorted(lengths)
                q95 = sorted_l[int(0.95 * (len(sorted_l)-1))] if sorted_l else 128
                dynamic_max_len = min(q95, self.max_seq_len or 4096)

            avg_len = sum(lengths) / len(lengths) if lengths else 0.0
            pct_truncated = (sum(l > dynamic_max_len for l in lengths) / len(lengths)) if lengths else 0.0

            # For multi-GPU with device_map='auto', inputs should be on CPU
            enc = self.tokenizer(combined, return_tensors="pt", padding=True, truncation=True,
                                max_length=dynamic_max_len)
            state_enc = self.tokenizer(obs, return_tensors="pt", padding=True, truncation=True,
                                    max_length=dynamic_max_len)


        return enc, state_enc, advs, rets, old_logps, label_masks, obs, avg_len, pct_truncated

    def _mini_batch_update_step(self, steps, scaling: float = 1.0):
        enc, state_enc, advs, rets, old_logps, label_masks, obs, avg_len, pct_truncated = self._prepare_batch(steps=steps)
        
        # 检查是否使用多GPU分布式模型
        has_device_map = hasattr(self.shared_model.base_model, 'hf_device_map') and self.shared_model.base_model.hf_device_map
        
        if has_device_map:
            # 多GPU模式：输入保持在CPU，让模型自动分发
            # 只移动非输入张量到主设备
            device = self.device
            advs = advs.to(device)
            rets = rets.to(device)
            # old_logps 和 label_masks 稍后根据模型输出设备动态调整
        else:
            # 单GPU模式：所有张量移到主设备
            device = self.device
            advs = advs.to(device)
            rets = rets.to(device)
            old_logps = old_logps.to(device)
            label_masks = label_masks.to(device)
            enc = tree.map_structure(lambda x: x.to(device), enc)
            state_enc = tree.map_structure(lambda x: x.to(device), state_enc)

        # 前向传播 - 策略部分
        policy_outputs = self.shared_model.get_policy_logits(enc.input_ids, enc.attention_mask)
        logp_full = torch.nn.functional.log_softmax(policy_outputs, dim=-1)          # [B, T, V]

        # 目标 token 对齐：去掉最后一位（因右移）
        tgt_ids  = enc.input_ids[:, 1:]                                          # [B, T-1]
        tok_logp = logp_full[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        
        # 确保old_logps和label_masks与模型输出在同一设备
        model_output_device = tok_logp.device
        old_logps = old_logps.to(model_output_device)
        label_mask = label_masks.to(model_output_device)

        # === 对齐长度：tok_logp / old_logps / label_masks 取共同最小 L ===
        L = min(tok_logp.size(1), old_logps.size(1), label_mask.size(1))
        tok_logp    = tok_logp[:, :L]
        old_logps   = old_logps[:, :L]
        label_mask  = label_mask[:, :L]        # 直接使用存档的 mask
        valid_tokens= label_mask.sum(dim=1).clamp_min(1)

        # === 序列 logp 与 熵 ===
        tok_logp_masked = tok_logp.masked_fill(~label_mask, 0.0)
        seq_logp  = tok_logp_masked.sum(dim=1) / valid_tokens

        probs = torch.nn.functional.softmax(policy_outputs, dim=-1)
        token_entropy = -(probs * logp_full).sum(-1)[:, :-1][:, :L]     # [B, L]
        seq_entropy   = token_entropy.masked_fill(~label_mask, 0.0).sum(dim=1) / valid_tokens

        # === 数值保底 ===
        seq_logp    = torch.nan_to_num(seq_logp,    nan=0.0, posinf=30.0, neginf=-30.0)
        old_logps   = torch.nan_to_num(old_logps,   nan=0.0, posinf=30.0, neginf=-30.0)
        seq_entropy = torch.nan_to_num(seq_entropy, nan=0.0, posinf=30.0, neginf=0.0)

        # （可选）mini-batch 级优势标准化
        if getattr(self, "normalize_adv", False) and not getattr(self, "_already_normalized", False):
            advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)

        # === ratio / clipped objective ===
        token_ratio_logits = (tok_logp - old_logps).masked_fill(~label_mask, 0.0)  # [B, L]
        seq_ratio_logits   = (token_ratio_logits.sum(dim=1) / valid_tokens).clamp(-20.0, 20.0)
        ratio = torch.exp(seq_ratio_logits)

        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advs
        policy_loss = -torch.min(surr1, surr2).mean()

        # 熵正则
        entropy_loss = -seq_entropy.mean()

        # === KL 惩罚 ===
        token_kl = (old_logps - tok_logp).masked_fill(~label_mask, 0.0)            # [B, L]
        kl_div   = (token_kl.sum(dim=1) / valid_tokens).mean()
        kl_penalty = self.kl_coef * torch.clamp(kl_div - self.kl_target, min=0.0)

        # ---- Policy loss 反传 ----
        total_policy_loss = (policy_loss + self.entropy_coef * entropy_loss + kl_penalty) / scaling
        total_policy_loss.backward()
        torch.cuda.empty_cache()

        # ---- 值函数 ----
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            value_pred = self.shared_model.get_values(state_enc.input_ids, state_enc.attention_mask)
        value_loss = self.value_loss_coef * 0.5 * ((value_pred.float() - rets) ** 2).mean() / scaling
        value_loss.backward()
        torch.cuda.empty_cache()

        # 统计
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
            "value_mae": (value_pred.float() - rets).abs().mean().item(),
            "value_dir_acc": ((value_pred > 0) == (rets > 0)).float().mean().item(),
            "num_steps": len(steps),
            "avg_train_len": avg_len,
            "pct_truncated": pct_truncated,
        }


    def _update(self, batch):
        all_samples = tree.flatten(batch)
        num_samples = len(all_samples)
        all_values, all_old_logps, all_label_masks = [], [], []

       # 固定一次 snapshot 的 pad 长度，确保 old_logp/label_mask 全流程宽度一致
        self._snapshot_pad_len = self.max_train_len or self.max_seq_len or 4096

        for i in range(0, len(all_samples), self.infer_mini_batch_size):
            batch_steps = all_samples[i : i + self.infer_mini_batch_size]

            # 直接用固定长度编码，保证每个 mini-chunk 的宽度一致
            combined_texts = [s.obs + s.act for s in batch_steps]
            obs_texts      = [s.obs for s in batch_steps]

            enc = self.tokenizer(
                combined_texts, return_tensors="pt",
                padding="max_length", truncation=True, max_length=self._snapshot_pad_len
            )   # 保持CPU

            state_enc = self.tokenizer(
                obs_texts, return_tensors="pt",
                padding="max_length", truncation=True, max_length=self._snapshot_pad_len
            )   # 保持CPU

            with torch.no_grad():
                # 值函数
                target_values = self.shared_model.get_values(state_enc.input_ids, state_enc.attention_mask)
                all_values.append(target_values.cpu())

                # 旧策略 token-level logp（与上面 enc 严格对齐）
                policy_logits = self.shared_model.get_policy_logits(enc.input_ids, enc.attention_mask)
                logp = torch.nn.functional.log_softmax(policy_logits, dim=-1)           # [B, T, V]
                tgt_ids = enc.input_ids[:, 1:]                                          # [B, T-1]
                tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)# [B, T-1]

                # 生成与 snapshot 一致口径的 label_mask（非 pad 且 非 prompt）
                attn = enc.attention_mask.bool().cpu()     # [B, T] 强制在CPU
                prompt_lens = torch.tensor(
                    [len(self.tokenizer(o, add_special_tokens=False)["input_ids"]) for o in obs_texts],
                    device="cpu"                           # 强制CPU
                )
                token_idx = torch.arange(attn.size(1), device="cpu").unsqueeze(0)  # 强制CPU
                prompt_mask = token_idx < prompt_lens.unsqueeze(1)
                label_mask_tokens = (attn & (~prompt_mask))[:, 1:]

                all_old_logps.append(tok_logp.detach().cpu())          # 形状固定为 [B, L]，L = pad_len-1
                all_label_masks.append(label_mask_tokens.detach().cpu())  # 同上


        if not all_values:
            self.logger.warning("No values to concatenate, skipping update")
            return {"error": "No values to process", "step": self._step, "samples_seen": self._samples_seen}

        all_values       = torch.cat(all_values).float().cpu()            # [N]
        all_old_logps    = torch.cat(all_old_logps).float().cpu()         # [N, L]
        all_label_masks  = torch.cat(all_label_masks).bool().cpu()        # [N, L]

        ep_lens = [len(ep) for ep in batch]

        ep_values       = torch.split(all_values, ep_lens)
        ep_old_logps    = torch.split(all_old_logps, ep_lens)
        ep_label_masks  = torch.split(all_label_masks, ep_lens)

                
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
            # Check if we have advantages for this episode
            if i >= len(ep_advantages):
                self.logger.warning(f"Episode {i} exceeds available advantages (len={len(ep_advantages)}), skipping")
                continue
                
            for j, step in enumerate(ep):
                # Check if we have advantages for this step
                if j >= len(ep_advantages[i]):
                    self.logger.warning(f"Step {j} in episode {i} exceeds available advantages (len={len(ep_advantages[i])}), skipping")
                    continue
                    
                step = replace(step, reward=ep_advantages[i][j].item())
                # 存储token-level的old_logp
                if j < len(ep_old_logps[i]):
                    old_logp_tokens = ep_old_logps[i][j].tolist() if hasattr(ep_old_logps[i][j], 'tolist') else [ep_old_logps[i][j].item()]
                else:
                    old_logp_tokens = [0.0]  # 默认值
                    
                step = replace(step, step_info={
                    **step.step_info,
                    "return":    ep_returns[i][j].item(),
                    "advantage": ep_advantages[i][j].item(),
                    "old_logp":  ep_old_logps[i][j].tolist(),       # [L]
                    "label_mask": ep_label_masks[i][j].tolist(),    # [L]（bool -> list）
                })

                train_batch.append(step)
        # Check if we have enough samples
        _restore_bs = None
        _restore_mbs = None
        if len(train_batch) < self.batch_size:
            self.logger.warning(f"Not enough samples in train_batch: {len(train_batch)} < {self.batch_size}")
            # 如果太少，直接返回；否则临时缩小本轮 batch_size
            if len(train_batch) < max(2, self.batch_size // 4):
                return {"error": "Not enough samples", "step": self._step, "samples_seen": self._samples_seen}
            _restore_bs = self.batch_size
            self.batch_size = len(train_batch)

        # 确保 mini-batch 不超过 batch-size；必要时临时下调
        if self.mini_batch_size > self.batch_size:
            _restore_mbs = self.mini_batch_size
            # 这里直接设成 batch_size（或至少 1）
            self.mini_batch_size = max(1, self.batch_size)

        if self.normalize_adv:
            train_batch = NormalizeRewardsByEnv(True)(train_batch)
            self._already_normalized = True

        # PPO多轮更新
        metrics_acc = {}
        # 随机采样到当前 batch_size
        train_batch = random.sample(train_batch, self.batch_size)

        # 使用“向上取整”的步数计算，避免 num_steps 为 0
        effective_mini_batch_size = max(1, min(self.mini_batch_size, self.batch_size))
        num_steps = max(1, (self.batch_size + effective_mini_batch_size - 1) // effective_mini_batch_size)

        self.logger.info(
            f"Got {num_samples} samples. Running PPO for {self.ppo_epochs} epochs, "
            f"{num_steps} steps per epoch (mini_batch_size={effective_mini_batch_size}, batch_size={self.batch_size})"
        )

        for epoch in range(self.ppo_epochs):
            self.policy_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)

            random.shuffle(train_batch)
            epoch_metrics = {}

            for i in range(num_steps):
                start = i * effective_mini_batch_size
                end   = min((i + 1) * effective_mini_batch_size, len(train_batch))
                sub = train_batch[start:end]
                if not sub:
                    continue

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    update_metrics = self._mini_batch_update_step(sub, scaling=num_steps)
                for k, v in update_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                    metrics_acc[k]   = metrics_acc.get(k, 0.0) + v

            # 梯度裁剪和优化器步骤 - 分别处理策略和价值参数
            torch.nn.utils.clip_grad_norm_(self.shared_model.base_model.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.shared_model.value_head.parameters(), self.grad_clip)

            try:
                self.policy_optimizer.step()
                self.critic_optimizer.step()
            except Exception as exc:
                self.logger.exception(f"optimizer.step crashed on step {self._step}, epoch {epoch} - {exc}")
                # 退出前恢复 batch/mini-batch
                if _restore_bs is not None:
                    self.batch_size = _restore_bs
                if _restore_mbs is not None:
                    self.mini_batch_size = _restore_mbs
                raise

            # 早停检查：如果KL散度太大，提前停止（放宽阈值）
            avg_kl = epoch_metrics.get("kl_div", 0.0) / max(1, num_steps)
            if avg_kl > 3.0 * self.kl_target:
                self.logger.info(f"Early stopping at epoch {epoch} due to high KL divergence: {avg_kl:.4f}")
                break

            # 打印更可读的字典
            self.logger.info(f"Epoch {epoch} metrics: { {k: (v / max(1, num_steps)) for k, v in epoch_metrics.items()} }")

        self._step += 1

        # 计算平均指标
        total_updates = max(1, (epoch + 1) * num_steps)
        log = {f"{k}": v / total_updates for k, v in metrics_acc.items()}

        # 计算梯度范数（跨设备安全）：在Python端聚合平方和，避免跨GPU stack
        policy_grads = [p.grad for p in self.shared_model.base_model.parameters() if p.grad is not None]
        critic_grads = [p.grad for p in self.shared_model.value_head.parameters() if p.grad is not None]
        if policy_grads:
            policy_sq_sum = 0.0
            for g in policy_grads:
                # 使用float32做平方和，取item避免设备冲突
                policy_sq_sum += g.detach().float().pow(2).sum().item()
            grad_norm = policy_sq_sum ** 0.5
        else:
            grad_norm = 0.0

        if critic_grads:
            critic_sq_sum = 0.0
            for g in critic_grads:
                critic_sq_sum += g.detach().float().pow(2).sum().item()
            critic_grad_norm = critic_sq_sum ** 0.5
        else:
            critic_grad_norm = 0.0

        log.update({
            "step": self._step,
            "samples_seen": self._samples_seen,
            "lr": self.policy_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "ppo_epochs_completed": epoch + 1,
        })

        # ✅ 正常路径也要恢复本轮临时缩小的 batch / mini-batch
        if _restore_bs is not None:
            self.batch_size = _restore_bs
        if _restore_mbs is not None:
            self.mini_batch_size = _restore_mbs

        # 重置归一化标志，避免下轮被误判“已归一化”
        self._already_normalized = False
        self._snapshot_pad_len = None
        return log

        