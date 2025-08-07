import ray, torch, tree, random, math
from typing import Optional, List, Dict, Any
from dataclasses import replace
from unstable.learners.base import BaseLearner
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt
from unstable.reward_transformations.transformation_sampling import NormalizeRewardsByEnv


def compute_sequence_importance_ratio(old_logprobs, new_logprobs, response_lengths, normalize_length=True):
    """
    Compute sequence-level importance ratio for GSPO
    Args:
        old_logprobs: log probabilities from old policy
        new_logprobs: log probabilities from new policy  
        response_lengths: lengths of responses for normalization
        normalize_length: whether to apply length normalization
    """
    if normalize_length:
        # Length normalization to reduce variance
        old_logprobs = old_logprobs / response_lengths
        new_logprobs = new_logprobs / response_lengths
    
    # Sequence-level importance ratio
    importance_ratio = torch.exp(new_logprobs - old_logprobs)
    return importance_ratio


def compute_group_advantages(rewards, group_size):
    """
    Compute group relative advantages for GSPO
    Args:
        rewards: tensor of rewards for all responses
        group_size: number of responses per query
    """
    batch_size = rewards.size(0)
    
    # Handle case where batch_size is not divisible by group_size
    if batch_size % group_size != 0:
        # Pad with zeros to make it divisible
        padding_size = group_size - (batch_size % group_size)
        padding = torch.zeros(padding_size, device=rewards.device, dtype=rewards.dtype)
        rewards_padded = torch.cat([rewards, padding], dim=0)
        
        # Reshape into groups
        rewards_grouped = rewards_padded.view(-1, group_size)
        
        # Compute group mean as baseline (only for non-padded elements)
        group_means = rewards_grouped.mean(dim=1, keepdim=True)
        
        # Compute advantages relative to group mean
        advantages_grouped = rewards_grouped - group_means
        
        # Flatten and remove padding
        advantages = advantages_grouped.flatten()[:batch_size]
    else:
        # Normal case: batch_size is divisible by group_size
        rewards_grouped = rewards.view(-1, group_size)
        group_means = rewards_grouped.mean(dim=1, keepdim=True)
        advantages = (rewards_grouped - group_means).flatten()
    
    return advantages


@ray.remote
class GSPOLearner(BaseLearner):
    def initialize_algorithm(self, 
                           infer_mini_batch_size: int, 
                           group_size: int = 4,
                           clip_ratio: float = 0.2,
                           normalize_length: bool = True,
                           normalize_adv: bool = False, 
                           max_generation_len: Optional[int] = None, 
                           max_train_len: Optional[int] = None, 
                           initial_lora_path: Optional[str] = None):
        """
        Initialize GSPO algorithm parameters
        """
        self.infer_mini_batch_size = infer_mini_batch_size
        self.group_size = group_size
        self.clip_ratio = clip_ratio
        self.normalize_length = normalize_length
        self.normalize_adv = normalize_adv
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len

        # GSPO doesn't need a separate critic model like A2C
        # It uses group-based advantage estimation instead
        
        if not self.use_trainer_cache:      
            self.policy_model.config.use_cache = False
        if self.gradient_checkpointing:     
            self.policy_model.gradient_checkpointing_enable()
        if self.activation_checkpointing:   
            enable_full_activation_ckpt(self.policy_model)

    def _prepare_batch(self, steps):
        """
        Prepare batch data for GSPO training
        """
        obs, acts, rewards = zip(*[(s.obs, s.act, s.reward) for s in steps])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # Combine observations and actions for full sequences
        combined = [o + a for o, a in zip(obs, acts)]
        
        # Calculate sequence lengths for normalization
        lengths = [len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in combined]
        response_lengths = torch.tensor([len(self.tokenizer(a, add_special_tokens=False)["input_ids"]) 
                                       for a in acts], dtype=torch.float32, device=self.device)
        
        avg_len = sum(lengths) / len(lengths)
        pct_truncated = (sum(l > self.max_train_len for l in lengths) / len(lengths) if self.max_train_len else 0)
        
        # Tokenize sequences
        enc = self.tokenizer(combined, return_tensors="pt", padding=True, 
                           truncation=True, max_length=self.max_train_len).to(self.device)
        
        # Tokenize observations for prompt masking
        obs_enc = self.tokenizer(obs, return_tensors="pt", padding=True, 
                               truncation=True, max_length=self.max_train_len).to(self.device)
        
        return enc, obs_enc, rewards, response_lengths, obs, avg_len, pct_truncated

    def _compute_sequence_logprobs(self, enc, obs_enc):
        """
        Compute sequence-level log probabilities
        """
        # Forward pass through policy model
        with torch.no_grad():
            old_outputs = self.policy_model(**enc)
        
        # Current policy forward pass
        outputs = self.policy_model(**enc)
        
        # Compute log probabilities
        logprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        old_logprobs = torch.nn.functional.log_softmax(old_outputs.logits, dim=-1)
        
        # Get target token IDs (shifted by 1)
        target_ids = enc.input_ids[:, 1:]
        
        # Extract log probabilities for target tokens
        token_logprobs = logprobs[:, :-1, :].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        old_token_logprobs = old_logprobs[:, :-1, :].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        
        # Create mask to exclude prompt tokens (only response tokens)
        mask = torch.ones_like(enc.input_ids, dtype=torch.bool, device=self.device)
        for i, obs in enumerate(obs_enc.input_ids):
            prompt_len = (obs != self.tokenizer.pad_token_id).sum().item()
            mask[i, :prompt_len] = False
        mask = mask[:, 1:]  # Shift mask to match target_ids
        
        # Sum log probabilities over response tokens only
        seq_logprobs = (token_logprobs * mask).sum(dim=1)
        old_seq_logprobs = (old_token_logprobs * mask).sum(dim=1)
        
        return seq_logprobs, old_seq_logprobs

    def _mini_batch_update_step(self, steps, scaling: float = 1.0):
        """
        Perform one mini-batch update step for GSPO
        """
        enc, obs_enc, rewards, response_lengths, obs, avg_len, pct_truncated = self._prepare_batch(steps)
        
        # Compute sequence-level log probabilities
        seq_logprobs, old_seq_logprobs = self._compute_sequence_logprobs(enc, obs_enc)
        
        # Compute sequence-level importance ratios
        importance_ratios = compute_sequence_importance_ratio(
            old_seq_logprobs, seq_logprobs, response_lengths, self.normalize_length
        )
        
        # Compute group-based advantages
        advantages = compute_group_advantages(rewards, self.group_size)
        advantages = advantages.to(self.device)
        
        # GSPO sequence-level clipped objective
        clipped_ratios = torch.clamp(importance_ratios, 
                                   1 - self.clip_ratio, 
                                   1 + self.clip_ratio)
        
        # Policy loss with sequence-level clipping
        policy_loss1 = advantages * importance_ratios
        policy_loss2 = advantages * clipped_ratios
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean() / scaling
        
        # Backward pass
        policy_loss.backward()
        torch.cuda.empty_cache()
        
        # Compute metrics
        clipped_fraction = (importance_ratios != clipped_ratios).float().mean()
        
        return {
            "policy_loss": policy_loss.item(),
            "importance_ratio_mean": importance_ratios.mean().item(),
            "importance_ratio_std": importance_ratios.std().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "clipped_fraction": clipped_fraction.item(),
            "num_steps": len(steps),
            "avg_train_len": avg_len,
            "pct_truncated": pct_truncated,
        }

    def _update(self, batch):
        """
        Main update function for GSPO
        """
        all_samples = tree.flatten(batch)
        num_samples = len(all_samples)
        
        # Group samples for group-based advantage computation
        # Ensure samples are properly grouped (group_size responses per query)
        if num_samples % self.group_size != 0:
            # Pad or truncate to make divisible by group_size
            target_size = (num_samples // self.group_size) * self.group_size
            all_samples = all_samples[:target_size]
            num_samples = target_size
        
        # Compute group-based advantages
        rewards = [step.reward for step in all_samples]
        group_advantages = compute_group_advantages(torch.tensor(rewards), self.group_size)
        
        # Update steps with computed advantages
        train_batch = []
        for i, step in enumerate(all_samples):
            step = replace(step, reward=group_advantages[i].item())
            step = replace(step, step_info={**step.step_info, "advantage": group_advantages[i].item()})
            train_batch.append(step)
        
        # Ensure we have enough samples for training
        if len(train_batch) < self.batch_size:
            self.logger.warning(f"Not enough samples for training: {len(train_batch)} < {self.batch_size}")
            return {"policy_loss": 0.0, "num_steps": 0}
        
        # Reset gradients
        self.policy_optimizer.zero_grad(set_to_none=True)
        
        # Sample training batch
        train_batch = random.sample(train_batch, self.batch_size)
        num_steps = self.batch_size // self.mini_batch_size
        
        self.logger.info(f"Got {num_samples} samples. Running {num_steps} mini-batch steps (mini_batch_size: {self.mini_batch_size})")
        
        # Apply advantage normalization if requested
        if self.normalize_adv:
            train_batch = NormalizeRewardsByEnv(True)(train_batch)
        
        # Accumulate metrics
        metrics_acc: Dict[str, float] = {}
        
        # Mini-batch training loop
        for i in range(num_steps):
            sub_batch = train_batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                update_metrics = self._mini_batch_update_step(sub_batch, scaling=num_steps)
            
            # Accumulate metrics
            for k, v in update_metrics.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v
            
            self.logger.info(f"Mini-step {i+1}/{num_steps} metrics: {update_metrics}")
        
        self.logger.info(f"Step accumulated metrics: {metrics_acc}")
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)
        
        # Optimizer step
        try:
            self.policy_optimizer.step()
        except Exception as exc:
            self.logger.exception(f"Optimizer step failed on step {self._step}: {exc}")
            raise
        
        self._step += 1
        
        # Prepare final metrics
        log = {f"{k}": v / num_steps for k, v in metrics_acc.items()}
        grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.policy_model.parameters() 
                        if p.grad is not None) ** 0.5)
        
        log.update({
            "step": self._step,
            "samples_seen": self._samples_seen,
            "lr": self.policy_optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
        })
        
        return log
