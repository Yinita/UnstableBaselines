# import time 
# from typing import List, Tuple, Dict, Optional

# import ray

# from transformers import AutoModelForCausalLM, AutoTokenizer

# # local imports
# from unstable.trajectory_buffer import StepBuffer
# from unstable.tracker import Tracker
# from unstable.collector import Collector
# from unstable.algorithms import BaseAlgo
# from unstable.utils.lora import load_lora_state

import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import ray, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from unstable.algorithms import BaseAlgo


def _build_peft_model(base_name: str, device: torch.device, lora_cfg: Dict[str, Any] | None, initial_lora_path: Optional[str], freeze_base: bool = True):
    """Load base model + wrap with LoRA.  Return (model, tokenizer)."""
    print(f"[Learner] Loading base model: {base_name} …")
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    if freeze_base:
        for p in base.parameters(): p.requires_grad_(False)

    lcfg = LoraConfig(
        r=lora_cfg.get("lora_rank", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    model = get_peft_model(base, lcfg).to(device)

    if initial_lora_path:
        print(f"[Learner] Loading initial LoRA weights from {initial_lora_path}")
        load_lora_state(model, initial_lora_path)

    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token  # safe default
    return model, tok




@ray.remote
class Learner:
    def __init__(
        self,
        model_name: str,
        iterations: int,
        step_buffer,  # StepBuffer handle (Ray actor)
        model_pool,
        algorithm,
        batch_size: int = 384,
        gradient_accum_steps: int = 32,
        learning_rate: float = 5e-6,
        grad_clip: float = 1.0,
        batch_delay_buffer: float = 1.5,
        lora_cfg: Optional[Dict[str, Any]] = None,
        initial_lora_path: Optional[str] = None,
        num_learners: int = 1,
        ckpt_root: str = "checkpoints",
        save_every: int = 50,
        tracker=None,  # WandBTracker handle (optional)
    ):
        """Async learner.

        Args
        -----
        iterations : number of optimisation *steps* (gradient steps) to run.
        batch_size : *total* batch size per optimisation step.
        gradient_accum_steps : split batch into this many micro‑batches.
        batch_delay_buffer : how much larger the buffer has to be (× batch)
            before the first optimisation step – helps avoid early bias.
        """
        self.iterations = iterations
        self.batch_size = batch_size
        self.grad_accum = gradient_accum_steps
        self.grad_clip = grad_clip
        self.batch_delay_buffer = batch_delay_buffer
        self.step_buffer = step_buffer
        self.model_pool = model_pool
        self.tracker = tracker
        self.save_every = save_every
        self.ckpt_root = Path(ckpt_root)
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

        # — device selection —
        gpu_ids = ray.get_gpu_ids()
        self.device = (
            torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")
        )

        # — build model & algo —
        lora_cfg = lora_cfg or {}
        self.model, self.tokenizer = _build_peft_model(
            model_name, self.device, lora_cfg, initial_lora_path
        )

        if num_learners > 1:
            # Wrap in DDP (assumes learners launched with 1 GPU each)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[0], output_device=0, find_unused_parameters=False)

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        self.algorithm.initialize(model=self.model, tokenizer=self.tokenizer, device=self.device)

        # training counters
        self._step = 0  # optimisation step counter
        self._samples_seen = 0


    def train(self):
        print("[Learner] 🚀 starting training loop …")
        mini_bs = self.batch_size // self.grad_accum

        while self._step < self.iterations:
            while (ray.get(self.step_buffer.size.remote()) < self.batch_size * self.batch_delay_buffer): time.sleep(0.2)

            batch: List = ray.get(self.step_buffer.get_batch.remote(self.batch_size))
            assert len(batch) == self.batch_size
            self._samples_seen += len(batch)

            metrics_acc: Dict[str, float] = {}
            for i in range(self.grad_accum):
                sub = batch[i * mini_bs : (i + 1) * mini_bs]
                update_metrics = self.algorithm.update(sub, scaling=self.grad_accum)
                for k, v in update_metrics.items():
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + v

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._step += 1

            # Logging
            if self.tracker is not None:
                log = {f"learner/{k}": v / self.grad_accum for k, v in metrics_acc.items()}
                log.update({"learner/step": self._step, "learner/samples_seen": self._samples_seen, "learner/lr": self.optimizer.param_groups[0]["lr"]})
                self.tracker.log_learner.remote(log)
            else:
                if self._step % 10 == 0:
                    print(f"[Learner] step {self._step:>5} | loss={metrics_acc.get('loss', 0)/self.grad_accum:.4f}")

            # save & register checkpoint every step
            if self._step % self.save_every == 0:
                self._save_checkpoint()
                if self.model_pool and self._last_ckpt:
                    self.model_pool.add_checkpoint.remote(str(self._last_ckpt), self._step)
                    print(f"[Learner] ↪registered → {self._last_ckpt}")

        print("[Learner] training finished.")
        return {"final_step":self._step, "samples":self._samples}

    def _save_checkpoint(self):
        ckpt_dir = self.ckpt_root / f"iteration-{self._step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mdl = self.model.module if hasattr(self.model,'module') else self.model
        mdl.save_pretrained(ckpt_dir)
        self._last_ckpt = ckpt_dir
        print(f"[Learner] saved → {ckpt_dir}")

#     def _save_checkpoint(self):
#         ckpt_dir = self.ckpt_root / f"iteration-{self._step}"
#         ckpt_dir.mkdir(parents=True, exist_ok=True)
#         peft = (self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model)
#         peft.save_pretrained(ckpt_dir)
#         print(f"[Learner] saved checkpoint → {ckpt_dir}")

#         if self.model_pool is not None:
#             self.model_pool.add_checkpoint.remote(str(ckpt_dir), self._step)
# "}




# @ray.remote
# class Learner:
#     def __init__(
#         self,
#         model_name: str,
#         iterations: int,
#         num_learners: int = 1,
#         learning_rate: float = 1e-4,
#         gradient_clip: float = 1.0,
#         batch_size: int = 384,
#         gradient_accumulation_steps: int = 384,
#         batch_delay_buffer: float = 1.5, # an extra collection buffer to make sure we are not biasing the first training step too much
#         step_buffer: StepBuffer,
#         tracker: Tracker,
#         collector: Collector,
#         algorithm_class: BaseAlgo,
#         lora_cfg: Optional[Dict[str, Any]] = None,
#         initial_lora_path: Optional[str] = None
#     ):
#         self.model_name = model_name 
#         self.iterations = iterations
#         self.num_learners = num_learners

#         self.step_buffer = step_buffer 
#         self.tracker = tracker
#         self.collector = collector 

#         # learning params
#         self.gradient_clip = gradient_clip
#         self.batch_size = batch_size
#         self.batch_delay_buffer = batch_delay_buffer
#         self.gradient_accumulation_steps = gradient_accumulation_steps

#         self._build_model(lora_cfg=lora_cfg, initial_lora_path=initial_lora_path) # build the model
#         self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate) # optimizer over only the adapters
#         self.algorithm = algorithm_class(model=self.model, tokenizer=self.tokenizer, device=self.device) # initialize the algorithm

#         # determine mini-batch size
#         ctx = get_context()
#         rank = ctx.get_world_rank()
#         world_size = ctx.get_world_size()
#         self.gpu_batch_size = batch_size // world_size


#     def _build_model(self, lora_cfg: Dict[str, Any], initial_lora_path: Optional[str] = None):
#         local_gpu = rank % torch.cuda.device_count()
#         self.device = torch.device(f"cuda:{local_gpu}")

#         # load base + LoRA
#         base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
#         peft_model = build_lora_model(model=base, lora_cfg=lora_cfg).to(self.device) 
#         if initial_lora_path: load_lora_state(peft_model, initial_lora_path) # load weights if available
#         self.model = torch.nn.parallel.DistributedDataParallel(peft_model, device_ids=[local_gpu], output_device=local_gpu, find_unused_parameters=False)
#         self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)


#     def train(self):
#         for iteration in range(self.iterations):
#             # wait until data is available
#             while ray.get(self.buffer.size.remote()) < self.batch_size * self.batch_delay_buffer: time.sleep(0.2)

#             # pull batch
#             batch = ray.get(self.buffer.get_batch.remote(self.gpu_batch_size))
#             self.optimizer.zero_grad(set_to_none=True)

#             metrics = {}
#             mini_batch_size = len(batch)//self.gradient_accumulation_steps
#             for i in range(self.gradient_accumulation_steps):
#                 start, end = i*mini_batch_size, (i+1)*mini_batch_size # TODO adjust the scaling factor
#                 update_info = self.algorithm.update(steps=mini_batch, scaling=self.gradient_accumulation_steps)
#                 for k in update_info:
#                     metrics[k] = metrics.get(k, 0.0) + update_info[k]

#             # step
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
#             self.optimizer.step()

#             if rank == 0:
#                 # pass to wandb tracker for logging
#                 avg_metrics = {f"learner/{k}": v / self.gradient_accumulation_steps for k, v in metrics.items()}
#                 avg_metrics["learner/iteration"] = iteration
#                 avg_metrics["learner/grad_norm"] = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
#                 avg_metrics["learner/lr"] = optimizer.param_groups[0]["lr"]
#                 self.tracker.log_training_step(avg_metrics)

#                 # store checkpoint TODO
#                 checkpoint_folder_path = os.path.join(root_checkpoint_dir, f"iteration-{iteration}")
#                 os.makedirs(checkpoint_folder_path, exist_ok=True)
#                 peft_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
#                 peft_model.save_pretrained(checkpoint_folder_path)
#                 ray.get(collector.add_new_lora_paths.remote(checkpoint_folder_path, iteration))
#                 session.report({"iteration": iteration})

#         # let the collector know that we are done
#         if rank == 0: ray.get(self.collector.mark_done.remote())


