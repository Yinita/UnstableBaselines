
import time, pathlib, math, json, logging
from typing import Any, Dict, Optional, List

import ray, torch, transformers

# local imports
from unstable.buffer import StepBuffer
from unstable.core import BaseAlgo
from unstable.model_pool import ModelPool
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt
from unstable.utils.logging import setup_logger



@ray.remote
class StandardLearner:
    """
    Single-GPU learner that fine-tunes a PEFT LoRA adapter with the supplied `BaseAlgo`. Runs entirely on one Ray worker pinned to one GPU.
    * Uses mixed-precision bfloat16 by default.
    * `save_every` controls how often a LoRA checkpoint is written *and* registered with the `ModelPool`.
    """
    def __init__(
        self, model_name: str, step_buffer: StepBuffer, model_pool: ModelPool, algorithm: BaseAlgo, batch_size: int, mini_batch_size: int, learning_rate: float=1e-5,
        grad_clip: float=1.0, batch_delay_buffer: float=1.5, lora_cfg: Dict[str,Any] = {}, initial_lora_path: Optional[str]=None, num_learners: int=1, ckpt_root: str="checkpoints",
        save_every: int = 1, tracker=None, activation_checkpointing: bool=True, gradient_checkpointing: bool=True, use_trainer_cache: bool=False, max_train_len: Optional[int]=None
    ):
        """
        Initialise a single-GPU LoRA learner.

        Parameters
        ----------
        model_name (str): Hugging-Face hub ID or local path of the *base* model.
        step_buffer (StepBuffer): Remote buffer that supplies sampled `Step`s.
        model_pool (ModelPool): Remote actor responsible for checkpoint registration and opponent sampling. 
        algorithm (BaseAlgo): On-policy algorithm (e.g. `Reinforce`) that implements ``prepare_batch`` and ``update``.
        batch_size (int): Number of *Step* objects pulled from the buffer per learner step.
        mini_batch_size (int): Sub-batch size used for gradient accumulation. ``batch_size // mini_batch_size`` forward/backward passes are executed per learner step.
        learning_rate (float, default 1e-5): Peak LR for the LoRA parameters.
        grad_clip (float, default 1.0): Global-norm gradient clipping threshold.
        batch_delay_buffer (float, default 1.5): Start a learner step only when the buffer holds ``batch_size * batch_delay_buffer`` samples; smooths startup.
        lora_cfg (dict): ``{"lora_rank", "lora_alpha", "lora_dropout", "target_modules", …}``.
        initial_lora_path (str or None): Folder that already contains ``adapter_model.*`` to *warm start* from.
        num_learners (int, default 1): Informational only (helps dashboards when running multi-learner setups).
        ckpt_root (str, default ``"checkpoints"``): Fallback directory when no `Tracker` is provided.
        save_every (int, default 1): How many learner steps between checkpoint dumps & registration.
        tracker (Tracker or None): Remote tracker for logging losses, grad-norms, etc.
        activation_checkpointing (bool, default True): If ``True`` wrap every transformer block in a no-reentrant checkpoint wrapper (saves memory, costs compute).
        gradient_checkpointing (bool, default True): Use Hugging-Face's built-in per-layer grad-ckpt toggle.
        use_trainer_cache (bool, default False): Keep key/value cache during training (not recommended for LoRA-only updates).
        max_train_len (int or None): Truncate concatenated ``obs+act`` texts to this many tokens inside `algorithm.prepare_batch`.
        """
        self.logger = setup_logger("learner", ray.get(tracker.get_log_dir.remote())) # set up logging
        self.algorithm = algorithm
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size  
        self.grad_clip = grad_clip
        self.batch_delay_buffer = batch_delay_buffer
        self.step_buffer = step_buffer
        self.model_pool = model_pool
        self.tracker = tracker
        self.save_every = save_every
        self.ckpt_root = pathlib.Path(ray.get(self.tracker.get_checkpoints_dir.remote()) if self.tracker else ckpt_root)
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

        torch.set_float32_matmul_precision('high')
        torch.set_default_dtype(torch.bfloat16)

        gpu_ids = ray.get_gpu_ids()
        self.device = (torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu"))
        self.model, self.tokenizer = build_peft_model(model_name, self.device, lora_cfg, initial_lora_path)
        self.model.to(torch.bfloat16)

        if not use_trainer_cache:      self.model.config.use_cache = False
        if gradient_checkpointing:     self.model.gradient_checkpointing_enable() # gradient checkpointing
        if activation_checkpointing:   enable_full_activation_ckpt(self.model)
        
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        self.algorithm.initialize(model=self.model, tokenizer=self.tokenizer, device=self.device, max_train_len=max_train_len) # TODO create a tabel with recommended amounts for different vram qtys
        self._step = 0; self._samples_seen = 0 # training counters

    def train(self, iterations: int):
        """
        Run the main training loop. A *learner step* consists of:
        1. Waiting until the `StepBuffer` has at least ``batch_size * batch_delay_buffer`` samples.
        2. Pulling exactly `batch_size` Steps and splitting them into ``batch_size // mini_batch_size`` micro-batches.
        3. Calling ``algorithm.update`` under ``torch.autocast`` and accumulating gradients.
        4. Global-norm clipping and `optimizer.step()`.
        5. Optional checkpoint save + registration.

        Parameters
        ----------
        iterations (int): Number of learner steps to execute before returning.
        """
        self.logger.info("Learner starting training loop …")
        while self._step < iterations:
            while (ray.get(self.step_buffer.size.remote()) < self.batch_size * self.batch_delay_buffer): time.sleep(0.2)
            self.logger.info("Starting learning step")

            # batch: List = ray.get(self.step_buffer.get_batch.remote(self.batch_size))
            try:                        batch: List = ray.get(self.step_buffer.get_batch.remote(self.batch_size))
            except Exception as exc:    self.logger.exception(f"could not fetch batch (step={self._step}) -\n\n{exc}\n\n"); raise

            assert len(batch) == self.batch_size
            self._samples_seen += len(batch)
            self.optimizer.zero_grad(set_to_none=True)

            metrics_acc: Dict[str, float] = {}
            num_steps=self.batch_size//self.mini_batch_size
            self.logger.info(f"Got {len(batch)} many samples. Running for {num_steps} steps (i.e. mini batch size: {self.mini_batch_size})")
            
            for i in range(num_steps):
                sub = batch[i * self.mini_batch_size : (i + 1) * self.mini_batch_size]
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    update_metrics = self.algorithm.update(sub, scaling=num_steps)
                for k, v in update_metrics.items():
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + v
                self.logger.info(f"Mini-step metrics: {update_metrics}")
            self.logger.info(f"Step metrics: {metrics_acc}")
            

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            try:                        self.optimizer.step()
            except Exception as exc:    self.logger.exception(f"optimizer.step crashed on step {self._step} -\n\n{exc}\n\n"); raise
            self._step += 1

            # Logging
            if self.tracker is not None:
                log = {f"{k}": v / num_steps for k, v in metrics_acc.items()}
                log.update({
                    "step": self._step,  "samples_seen": self._samples_seen,  "lr": self.optimizer.param_groups[0]["lr"], 
                    "grad_norm": sum(p.grad.data.norm(2).item()**2 for p in self.model.parameters() if p.grad is not None) ** 0.5
                })
                self.tracker.log_learner.remote(log)
            else:
                if self._step % 10 == 0: 
                    self.logger.info(f"[Learner] step {self._step:>5} | loss={metrics_acc.get('loss', 0):.4f}")

            # save & register checkpoint every step
            if self._step % self.save_every == 0:
                try:                        self._save_checkpoint()
                except Exception as exc:    self.logger.exception(f"failed to save checkpoint {ckpt_dir} -\n\n{exc}\n\n"); raise
                # self._save_checkpoint()
                if self.model_pool and self._last_ckpt:
                    self.model_pool.add_checkpoint.remote(str(self._last_ckpt), self._step)
                    self.logger.info(f"[Learner] +registered -> {self._last_ckpt}")
                    self.model_pool.snapshot.remote(self._step)

        self.logger.info("[Learner] training finished.")
        self.step_buffer.stop()
        return {"final_step":self._step, "samples":self._samples}

    def _save_checkpoint(self):
        """
        Serialise the current LoRA adapter to ``<ckpt_root>/iteration-{step}``.
        * Uses `.save_pretrained(..., save_adapter=True)` to write only LoRA weights and config - the frozen base model is **not** duplicated.
        * Updates ``self._last_ckpt`` so the next call to `model_pool.add_checkpoint` knows where to find the files.
        """
        ckpt_dir = self.ckpt_root / f"iteration-{self._step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model = self.model.module if hasattr(self.model,'module') else self.model
        model.save_pretrained(ckpt_dir, save_adapter=True)
        self._last_ckpt = ckpt_dir
        self.logger.info(f"[Learner] saved -> {ckpt_dir}")

