import ray, torch
from typing import Optional
from unstable.learners.base import BaseLearner
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt




@ray.remote
class A2CLearner(BaseLearner):
    def initialize_algorithm(self, infer_mini_batch_size: int, critic_learning_rate: float, normalize_adv: bool=False, initial_lora_path: Optional[str]=None):
        self.infer_mini_batch_size = infer_mini_batch_size
        self.normalize_adv = normalize_adv

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

    def _mini_batch_update_step(self, batch):
        metrics_acc = {}
        self.policy_optimizer.zero_grad(set_to_none=True)
        for i in range(self.gradient_acc_steps):
            sub = batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16): 
                update_metrics = self._mini_batch_update_step.update(sub, scaling=self.gradient_acc_steps)
            for k, v in update_metrics.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v
            self.logger.info(f"Mini-step metrics: {update_metrics}")
        self.logger.info(f"Step metrics: {metrics_acc}")
        # update weights
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.grad_clip)
        self.policy_optimizer.step()
        return metrics_acc

    def _update(self, batch):
        all_samples = tree.flatten(batch_episodes)
        num_samples = len(all_samples)

        # construct value targets
        all_values = []
        for i in range(0, len(all_samples), self.infer_mini_batch_size):
            batch_steps = all_samples[i : i + self.infer_mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                _, state_enc, _, _, _, _, _ = self.algorithm.prepare_batch(batch_steps)
                with torch.no_grad():
                    # take the first token's output
                    target_values = self.critic(**state_enc)[:, 0]
                    all_values.append(target_values)
        all_values = torch.cat(all_values).float().cpu()
        ep_values = torch.split(all_values, [len(ep) for ep in batch_episodes])
        ep_rewards = [torch.tensor([step.reward for step in ep]) for ep in batch_episodes]
        ep_advantages = []
        ep_returns = []
        for rewards, values in zip(ep_rewards, ep_values):
            adv = compute_gae(rewards, values)
            ep_advantages.append(adv)
            ep_returns.append(adv + values)

        batch = []
        for i, ep in enumerate(batch_episodes):
            for j, step in enumerate(ep):
                step = replace(step, reward=ep_advantages[i][j].item())
                step = replace(step, step_info={**step.step_info, "return": ep_returns[i][j].item(), "advantage": ep_advantages[i][j].item()})
                batch.append(step)
        assert len(batch) >= self.batch_size

        self.policy_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

        metrics_acc: Dict[str, float] = {}

        batch = random.sample(batch, self.batch_size)
        num_steps = self.batch_size // self.mini_batch_size
        print(f"Got {num_samples} many samples. Running for {num_steps} steps (i.e. mini batch size: {self.mini_batch_size})")

        if self.normalize_adv: batch = NormalizeRewardsByEnv(True)(batch)
        for i in range(num_steps):
            sub = batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                update_metrics = self._update(sub, scaling=num_steps)
            for k, v in update_metrics.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v
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
        grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5)
        critic_grad_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in self.critic.parameters() if p.grad is not None)** 0.5)
        log.update({
            "step": self._step, "samples_seen": self._samples_seen, "lr": self.optimizer.param_groups[0]["lr"], 
            "grad_norm": grad_norm, "critic_lr": self.critic_optimizer.param_groups[0]["lr"], "grad_norm": critic_grad_norm,
        })








