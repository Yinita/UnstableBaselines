# # ─── learner/base_learner.py ───────────────────────────────────────────────
# import pathlib, time, ray, torch
# from typing import Any, Dict, List, Optional

# from unstable.learners.utils import (
#     build_peft_model,
#     enable_full_activation_ckpt,
# )
# from unstable.utils.logging import setup_logger


# class BaseLearner(ray.actor.Actor):        # plain class works too; Ray likes clarity
#     """
#     * Loads ONE “policy” backbone in __init__ (with all the fancy LoRA / ckpt tricks).
#     * `initialize_algorithm()` is a **hook** the subclass can override to:
#          – add extra models (critic, ref, etc.)
#          – allocate buffers
#          – pre-build schedulers / EMA shadows … whatever it needs.
#     * Training loop calls `_update(batch)` once per mini-batch.
#       The subclass provides that implementation.
#     """
#     # ------------------------------------------------------------------ #
#     #              1.  Construction & single policy model                #
#     # ------------------------------------------------------------------ #
#     def __init__(
#         self,
#         *,
#         step_buffer,              # BaseBuffer handle
#         model_pool,               # ModelPool handle
#         tracker,                  # Tracker handle
#         cfg,                      # simple dataclass with hyper-params
#     ):
#         # ----- logging / misc -----------------------------------------
#         self.tracker   = tracker
#         log_dir        = ray.get(tracker.get_log_dir.remote())
#         self.logger    = setup_logger("learner", log_dir)

#         # ----- load backbone (+ LoRA) ---------------------------------
#         gpu_ids        = ray.get_gpu_ids()
#         self.device    = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")

#         torch.set_default_dtype(torch.bfloat16)
#         torch.set_float32_matmul_precision("high")

#         self.policy, self.tokenizer = build_peft_model(
#             cfg.model_name,
#             self.device,
#             cfg.lora_cfg,
#             cfg.initial_lora_path,
#         )
#         self.policy.to(torch.bfloat16)

#         if cfg.gradient_ckpt:     self.policy.gradient_checkpointing_enable()
#         if cfg.activation_ckpt:   enable_full_activation_ckpt(self.policy)
#         if not cfg.use_trainer_cache:
#             self.policy.config.use_cache = False

#         # collect parameters that will be trained
#         self._trainable_params: List[torch.nn.Parameter] = list(
#             filter(lambda p: p.requires_grad, self.policy.parameters())
#         )

#         # ----- let the subclass inject anything else ------------------
#         #
#         #   Most subclasses will *either* leave this empty (REINFORCE),
#         #   *or* add a value head / critic model / reference model and
#         #   append their trainable parameters to `self._trainable_params`.
#         #
#         self.initialize_algorithm(cfg)

#         # ----- optimiser ---------------------------------------------
#         self.optim = torch.optim.AdamW(
#             self._trainable_params,
#             lr=cfg.learning_rate,
#         )

#         # ----- misc runtime state ------------------------------------
#         self.cfg          = cfg
#         self.step_buffer  = step_buffer
#         self.model_pool   = model_pool
#         self._step        = 0
#         self._samples     = 0
#         self._last_ckpt   = None

#     # ------------------------------------------------------------------ #
#     #              2.  Algorithm-specific set-up hook                     #
#     # ------------------------------------------------------------------ #
#     def initialize_algorithm(self, cfg):
#         """
#         Override in a subclass.

#         Typical things to do here:
#           • add `self.value_head = torch.nn.Linear(...)`
#           • deep-copy a second model and freeze it (`self.ref_model`)
#           • extend `self._trainable_params` with new parameters
#           • pre-compute masks, KL coefficients, etc.
#         """
#         pass                                                           # default → nothing

#     # ------------------------------------------------------------------ #
#     #                      3.  Training loop                              #
#     # ------------------------------------------------------------------ #
#     def train(self, iterations: int):
#         self.logger.info("[Learner] starting …")
#         batch_size      = self.cfg.batch_size
#         mini_batch_size = self.cfg.mini_batch_size
#         factor          = batch_size // mini_batch_size     # gradient-acc steps

#         while self._step < iterations:
#             # wait until buffer is “full enough”
#             needed = batch_size * self.cfg.batch_delay_buffer
#             while ray.get(self.step_buffer.size.remote()) < needed:
#                 time.sleep(0.2)

#             batch: List[Any] = ray.get(self.step_buffer.get_batch.remote(batch_size))
#             self._samples += len(batch)
#             self.optim.zero_grad(set_to_none=True)

#             metrics_acc: Dict[str, float] = {}
#             for i in range(factor):
#                 sub = batch[i * mini_batch_size : (i + 1) * mini_batch_size]
#                 with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
#                     m = self._update(sub, scaling=factor)   # ← subclass hook
#                 for k, v in m.items():
#                     metrics_acc[k] = metrics_acc.get(k, 0.0) + v

#             torch.nn.utils.clip_grad_norm_(self._trainable_params, self.cfg.grad_clip)
#             self.optim.step()

#             # ----- bookkeeping & tracker --------------------------------
#             log = {k: v / factor for k, v in metrics_acc.items()}
#             log.update(
#                 step=self._step,
#                 samples_seen=self._samples,
#                 lr=self.optim.param_groups[0]["lr"],
#             )
#             self.tracker.log_learner.remote(log)

#             # ----- checkpoint -------------------------------------------
#             if (self._step + 1) % self.cfg.save_every == 0:
#                 self._save_checkpoint()

#             self._step += 1

#         self.logger.info("[Learner] finished!")
#         self.step_buffer.stop.remote()

#     # ------------------------------------------------------------------ #
#     #                  4.  Algorithm-specific update                      #
#     # ------------------------------------------------------------------ #
#     def _update(self, mini_batch: List[Any], *, scaling: int) -> Dict[str, float]:
#         """
#         One gradient-accumulation micro-step on `mini_batch`.
#         MUST:
#             • call .backward()
#             • return a dict of metrics (loss, kl, etc.)
#         Default implementation raises so you never forget.
#         """
#         raise NotImplementedError("You need to implement _update() in a subclass!")

#     # ------------------------------------------------------------------ #
#     #                       5.  Checkpoint helper                         #
#     # ------------------------------------------------------------------ #
#     def _save_checkpoint(self):
#         ckpt_dir = pathlib.Path(self.cfg.ckpt_root) / f"iteration-{self._step}"
#         ckpt_dir.mkdir(parents=True, exist_ok=True)
#         self.policy.save_pretrained(ckpt_dir / "policy", save_adapter=True)

#         # save any extra trainable model the subclass added
#         if hasattr(self, "value_head"):
#             torch.save(self.value_head.state_dict(), ckpt_dir / "value_head.pt")
#         if hasattr(self, "critic"):
#             self.critic.save_pretrained(ckpt_dir / "critic", save_adapter=True)

#         self._last_ckpt = ckpt_dir
#         if self.model_pool:
#             self.model_pool.add_checkpoint.remote(str(ckpt_dir), self._step)
#             self.model_pool.snapshot.remote(self._step)
#         self.logger.info(f"[Learner] saved → {ckpt_dir}")




import time, pathlib, ray, torch
from typing import Any, Dict, Optional, List
import random

import tree
from dataclasses import replace
from unstable.buffer import EpisodeBuffer
from unstable.core import BaseAlgo
from unstable.model_pool import ModelPool
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt
from unstable.utils.logging import setup_logger
from unstable.reward_transformations.transformation_sampling import (
    NormalizeRewardsByEnv,
)


def compute_gae(rewards, values, gamma=1.0, gae_lambda=1.0):
    # Compute gae (for policy learning) and return (for critic learning)
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
        advantages[t] = lastgaelam = (
            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        )
    return advantages


@ray.remote
class A2CLearner:
    def __init__(
        self,
        model_name: str,
        episode_buffer: EpisodeBuffer,
        model_pool: ModelPool,
        algorithm: BaseAlgo,
        batch_size: int,
        mini_batch_size: int,
        infer_mini_batch_size: int,
        max_generation_len: int,
        normalize_adv: bool = False,
        learning_rate: float = 1e-5,
        critic_learning_rate: float = 1e-5,
        grad_clip: float = 1.0,
        batch_delay_buffer: float = 1.5,
        lora_cfg: Dict[str, Any] = {},
        initial_lora_path: Optional[str] = None,
        ckpt_root: str = "checkpoints",
        save_every: int = 1,
        tracker=None,
        activation_checkpointing: bool = True,
        gradient_checkpointing: bool = True,
        use_trainer_cache: bool = False,
        max_train_len: Optional[int] = None,
    ):
        self.logger = setup_logger(
            "learner", ray.get(tracker.get_log_dir.remote())
        )  # set up logging
        self.episode_buffer, self.model_pool, self.tracker = (
            episode_buffer,
            model_pool,
            tracker,
        )
        (
            self.batch_size,
            self.mini_batch_size,
            self.infer_mini_batch_size,
            self.grad_clip,
        ) = (batch_size, mini_batch_size, infer_mini_batch_size, grad_clip)
        self.algorithm = algorithm
        self.batch_delay_buffer = batch_delay_buffer
        self.save_every = save_every
        self.normalize_adv = normalize_adv
        self.ckpt_root = pathlib.Path(
            ray.get(self.tracker.get_checkpoints_dir.remote())
            if self.tracker
            else ckpt_root
        )
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

        torch.set_float32_matmul_precision("high")
        torch.set_default_dtype(torch.bfloat16)

        gpu_ids = ray.get_gpu_ids()
        self.device = (
            torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")
        )
        self.model, self.tokenizer = build_peft_model(
            model_name, self.device, lora_cfg, initial_lora_path
        )
        self.model.to(torch.bfloat16)
        self.critic, _ = build_peft_model(
            model_name, self.device, lora_cfg, initial_lora_path, critic_model=True
        )

        if not use_trainer_cache:
            self.model.config.use_cache = False
            self.critic.config.use_cache = False
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()  # gradient checkpointing
            self.critic.gradient_checkpointing_enable()  # gradient checkpointing
        if activation_checkpointing:
            enable_full_activation_ckpt(self.model)
            enable_full_activation_ckpt(self.critic)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.critic.parameters()),
            lr=critic_learning_rate,
        )

        self.algorithm.initialize(
            model=(self.model, self.critic),
            tokenizer=self.tokenizer,
            device=self.device,
            max_train_len=max_train_len,
            max_generation_len=max_generation_len,
        )
        self._step = 0
        self._samples_seen = 0  # training counters

    def train(self, iterations: int):
        self.logger.info("Learner starting training loop …")
        while self._step < iterations:
            while (
                ray.get(self.episode_buffer.size.remote())
                < self.batch_size * self.batch_delay_buffer
            ):
                time.sleep(0.2)
            self.logger.info("Starting learning step")

            try:
                batch_episodes: List = ray.get(
                    self.episode_buffer.get_batch.remote(self.batch_size)
                )
            except Exception as exc:
                self.logger.exception(
                    f"could not fetch batch (step={self._step}) -\n\n{exc}\n\n"
                )
                raise

            all_samples = tree.flatten(batch_episodes)
            num_samples = len(all_samples)

            # construct value targets
            all_values = []
            for i in range(0, len(all_samples), self.infer_mini_batch_size):
                batch_steps = all_samples[i : i + self.infer_mini_batch_size]
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    _, state_enc, _, _, _, _, _ = self.algorithm.prepare_batch(
                        batch_steps
                    )
                    with torch.no_grad():
                        # take the first token's output
                        target_values = self.critic(**state_enc)[:, 0]
                        all_values.append(target_values)
            all_values = torch.cat(all_values).float().cpu()
            ep_values = torch.split(all_values, [len(ep) for ep in batch_episodes])
            ep_rewards = [
                torch.tensor([step.reward for step in ep]) for ep in batch_episodes
            ]
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
                    step = replace(
                        step,
                        step_info={
                            **step.step_info,
                            "return": ep_returns[i][j].item(),
                            "advantage": ep_advantages[i][j].item(),
                        },
                    )
                    batch.append(step)
            assert len(batch) >= self.batch_size

            self.optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)

            metrics_acc: Dict[str, float] = {}

            batch = random.sample(batch, self.batch_size)
            num_steps = self.batch_size // self.mini_batch_size
            print(
                f"Got {num_samples} many samples. Running for {num_steps} steps (i.e. mini batch size: {self.mini_batch_size})"
            )

            if self.normalize_adv:
                batch = NormalizeRewardsByEnv(True)(batch)

            for i in range(num_steps):
                sub = batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    update_metrics = self.algorithm.update(sub, scaling=num_steps)
                for k, v in update_metrics.items():
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                self.logger.info(f"Mini-step metrics: {update_metrics}")
            self.logger.info(f"Step metrics: {metrics_acc}")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

            try:
                self.optimizer.step()
                self.critic_optimizer.step()
            except Exception as exc:
                self.logger.exception(
                    f"optimizer.step crashed on step {self._step} -\n\n{exc}\n\n"
                )
                raise
            self._step += 1

            log = {f"{k}": v / num_steps for k, v in metrics_acc.items()}
            grad_norm = (
                sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in self.model.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            critic_grad_norm = (
                sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in self.critic.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            log.update(
                {
                    "step": self._step,
                    "samples_seen": self._samples_seen,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
                    "grad_norm": critic_grad_norm,
                }
            )
            self.tracker.log_learner.remote(log)
            self._samples_seen += self.batch_size

            # save & register checkpoint every step
            if self._step % self.save_every == 0:
                try:
                    self._save_checkpoint()
                except Exception as exc:
                    self.logger.exception(
                        f"failed to save checkpoint {ckpt_dir} -\n\n{exc}\n\n"
                    )
                    raise
                if self.model_pool and self._last_ckpt:
                    self.model_pool.add_checkpoint.remote(
                        str(self._last_ckpt), self._step
                    )
                    self.logger.info(f"[Learner] +registered -> {self._last_ckpt}")
                    self.model_pool.snapshot.remote(self._step)

        self.logger.info("[Learner] training finished.")
        self.episode_buffer.stop.remote()

    def _save_checkpoint(self):
        ckpt_dir = self.ckpt_root / f"iteration-{self._step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.save_pretrained(ckpt_dir, save_adapter=True)
        self._last_ckpt = ckpt_dir
        self.logger.info(f"[Learner] saved -> {ckpt_dir}")










from typing import Optional

import torch

from unstable.core import BaseAlgo


class AdvantageActorCritic(BaseAlgo):
    def initialize(
        self,
        model,
        tokenizer,
        device,
        max_generation_len: int,
        max_train_len: Optional[int] = None,
    ):
        self.model, self.critic = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_train_len = max_train_len
        self.max_generation_len = max_generation_len

    def prepare_batch(self, steps):
        obs, acts, advs, rets = zip(
            *[
                (s.obs, s.act, s.reward, s.step_info.get("return", torch.nan))
                for s in steps
            ]
        )
        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        rets = torch.tensor(rets, dtype=torch.float32, device=self.device)
        combined = [o + a for o, a in zip(obs, acts)]
        lengths = [
            len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            for text in combined
        ]
        avg_len = sum(lengths) / len(lengths)
        pct_truncated = (
            sum(l > self.max_train_len for l in lengths) / len(lengths)
            if self.max_train_len
            else 0
        )
        enc = self.tokenizer(
            combined,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_train_len,
        ).to(
            self.device
        )  # Tokenize with truncation
        state_enc = self.tokenizer(
            obs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_train_len,
        ).to(
            self.device
        )  # Tokenize with truncation
        return enc, state_enc, advs, rets, obs, avg_len, pct_truncated

    def update(self, steps, scaling: float = 1.0):
        enc, state_enc, advs, rets, obs, avg_len, pct_truncated = self.prepare_batch(
            steps=steps
        )
        # Learn policy
        out = self.model(**enc)
        logp = torch.nn.functional.log_softmax(out.logits, dim=-1)
        tgt_ids = enc.input_ids[:, 1:]
        tok_logp = logp[:, :-1, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        mask = torch.ones_like(
            enc.input_ids, dtype=torch.bool, device=self.device
        )  # build prompt mask
        for i, o in enumerate(obs):
            mask[i, : len(self.tokenizer(o, add_special_tokens=False)["input_ids"])] = (
                False
            )
        mask = mask[:, 1:]
        seq_logp = (tok_logp * mask).sum(1) / self.max_generation_len
        loss = -(advs * seq_logp).mean() / scaling
        loss.backward()
        torch.cuda.empty_cache()

        # Learn value
        value_pred = self.critic(**state_enc)[:, 0]
        value_loss = 0.5 * ((value_pred - rets) ** 2).mean()
        value_loss.backward()
        torch.cuda.empty_cache()

        return {
            "policy_loss": loss.item(),
            "value_loss": value_loss.item(),
            "logp_mean": seq_logp.mean().item(),
            "logp_std": seq_logp.std().item(),
            "num_steps": len(steps),
            "avg_train_len": avg_len,
            "pct_truncated": pct_truncated,
        }























