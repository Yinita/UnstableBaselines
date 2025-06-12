
import os, time, pathlib, math, json
from typing import Any, Dict, Optional, List

import ray, torch, transformers
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, apply_activation_checkpointing

# local imports
from unstable.buffer import StepBuffer
from unstable.core import BaseAlgo
from unstable.model_pool import ModelPool
from unstable.learners.utils import build_peft_model, make_checkpointing_filter


import deepspeed
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology


@ray.remote
class DeepSpeedLearner:
    def __init__(
        self,
        model_name: str,
        step_buffer: StepBuffer,
        model_pool: ModelPool,
        algorithm: BaseAlgo,
        batch_size: int = 384,
        gradient_accum_steps: int = 32,
        learning_rate: float = 5e-6,
        grad_clip: float = 1.0,
        batch_delay_buffer: float = 1.5,
        lora_cfg: Dict[str, Any] = {},
        initial_lora_path: Optional[str] = None,
        num_learners: int = 1,
        ckpt_root: str = "checkpoints",
        save_every: int = 1,
        tracker=None,
    ):
        """Async learner.

        Args
        -----
        batch_size : *total* batch size per optimisation step.
        gradient_accum_steps : split batch into this many micro‑batches.
        batch_delay_buffer : how much larger the buffer has to be (× batch)
            before the first optimisation step – helps avoid early bias.
        """
        self.activation_checkpointing = False # TODO add to init
        self.gradient_checkpointing = True  # TODO add to init

        self.algorithm = algorithm
        self.batch_size = batch_size
        self.grad_accum = gradient_accum_steps
        self.grad_clip = grad_clip
        self.batch_delay_buffer = batch_delay_buffer
        self.step_buffer = step_buffer
        self.model_pool = model_pool
        self.tracker = tracker
        self.save_every = save_every
        ckpt_root = ray.get(self.tracker.get_checkpoints_dir.remote()) if self.tracker else ckpt_root
        self.ckpt_root = pathlib.Path(ckpt_root)
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

        # — device selection —
        gpu_ids = ray.get_gpu_ids()
        rank = 0
        world_size = len(gpu_ids)
        local_rank = gpu_ids.index(min(gpu_ids))  # assume Ray gives contiguous GPU ids

        torch.cuda.set_device(local_rank)
        # torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        init_file = "/tmp/ds_ray_init_shared5"
        if rank == 0 and os.path.exists(init_file):
            os.remove(init_file)

        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size
        )


        self.device = torch.device(f"cuda:{local_rank}")


        ds_config_path = "unstable/learners/deepspeed_config.json"
        ds_config = json.load(open(ds_config_path))

        # self.device = (torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu"))

        self.model, self.tokenizer = build_peft_model(model_name, self.device, lora_cfg, initial_lora_path)
        # self.model.to(torch.bfloat16)
        self.model.gradient_checkpointing_enable()
        # if num_learners > 1: self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[0], output_device=0, find_unused_parameters=False) # Wrap in DDP (assumes learners launched with 1 GPU each)
        
        # self.model.config.use_cache = False
        # if self.gradient_checkpointing:
        #     self.model.gradient_checkpointing_enable() # gradient checkpointing
        
        # # if self.activation_checkpointing: # TODO fix the qwen3 hardcoding part. Can prob get class from model 
        # check_fn = make_checkpointing_filter(percentage=1.0, block_class=transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer)
        # apply_activation_checkpointing(self.model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
    
        # algorithm and optimizer
        # self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=filter(lambda p: p.requires_grad, self.model.parameters()),
            config_params=ds_config
        )


        self.algorithm.initialize(model=self.model, tokenizer=self.tokenizer, device=self.device, accelerator=self.model_engine)

        # training counters
        self._step = 0
        self._samples_seen = 0


    def train(self, iterations: int):
        print("[Learner] starting training loop …")
        mini_bs = self.batch_size // self.grad_accum

        while self._step < iterations:
            while (ray.get(self.step_buffer.size.remote()) < self.batch_size * self.batch_delay_buffer): time.sleep(0.2)

            batch: List = ray.get(self.step_buffer.get_batch.remote(self.batch_size))
            assert len(batch) == self.batch_size
            self._samples_seen += len(batch)
            self.optimizer.zero_grad(set_to_none=True)

            metrics_acc: Dict[str, float] = {}
            for i in range(self.grad_accum):
                sub = batch[i * mini_bs : (i + 1) * mini_bs]
                update_metrics = self.algorithm.update(sub, scaling=self.grad_accum)
                for k, v in update_metrics.items():
                    metrics_acc[k] = metrics_acc.get(k, 0.0) + v

            self.model_engine.step()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # self.optimizer.step()
            self._step += 1

            # Logging
            if self.tracker is not None:
                log = {f"{k}": v / self.grad_accum for k, v in metrics_acc.items()}
                log.update({
                    "step": self._step,  "samples_seen": self._samples_seen,  "lr": self.optimizer.param_groups[0]["lr"], 
                    "grad_norm": sum(p.grad.data.norm(2).item()**2 for p in self.model.parameters() if p.grad is not None) ** 0.5
                })
                self.tracker.log_learner.remote(log)
            else:
                if self._step % 10 == 0: print(f"[Learner] step {self._step:>5} | loss={metrics_acc.get('loss', 0)/self.grad_accum:.4f}")

            # save & register checkpoint every step
            if self._step % self.save_every == 0:
                self._save_checkpoint()
                if self.model_pool and self._last_ckpt:
                    self.model_pool.add_checkpoint.remote(str(self._last_ckpt), self._step)
                    print(f"[Learner] ↪registered → {self._last_ckpt}")
                    self.model_pool.snapshot.remote(self._step)

        print("[Learner] training finished.")
        return {"final_step":self._step, "samples":self._samples}

    def _save_checkpoint(self):
        ckpt_dir = self.ckpt_root / f"iteration-{self._step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model = self.model.module if hasattr(self.model,'module') else self.model
        model.save_pretrained(ckpt_dir, save_adapter=True)
        self._last_ckpt = ckpt_dir
        print(f"[Learner] saved → {ckpt_dir}")


# import time, pathlib, math, json
# from typing import Any, Dict, Optional, List

# import ray, torch, transformers
# from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, apply_activation_checkpointing

# # local imports
# from unstable.buffer import StepBuffer
# from unstable.core import BaseAlgo
# from unstable.model_pool import ModelPool
# from unstable.learners.utils import build_peft_model, make_checkpointing_filter

# from accelerate import Accelerator


# @ray.remote
# class Learner:
#     def __init__(
#         self,
#         model_name: str,
#         step_buffer: StepBuffer,
#         model_pool: ModelPool,
#         algorithm: BaseAlgo,
#         batch_size: int = 384,
#         gradient_accum_steps: int = 32,
#         learning_rate: float = 5e-6,
#         grad_clip: float = 1.0,
#         batch_delay_buffer: float = 1.5,
#         lora_cfg: Dict[str, Any] = {},
#         initial_lora_path: Optional[str] = None,
#         num_learners: int = 1,
#         ckpt_root: str = "checkpoints",
#         save_every: int = 1,
#         tracker=None,
#     ):
#         """Async learner.

#         Args
#         -----
#         batch_size : *total* batch size per optimisation step.
#         gradient_accum_steps : split batch into this many micro‑batches.
#         batch_delay_buffer : how much larger the buffer has to be (× batch)
#             before the first optimisation step – helps avoid early bias.
#         """
#         self.activation_checkpointing = False # TODO add to init
#         self.gradient_checkpointing = True  # TODO add to init

#         self.algorithm = algorithm
#         self.batch_size = batch_size
#         self.grad_accum = gradient_accum_steps
#         self.grad_clip = grad_clip
#         self.batch_delay_buffer = batch_delay_buffer
#         self.step_buffer = step_buffer
#         self.model_pool = model_pool
#         self.tracker = tracker
#         self.save_every = save_every
#         ckpt_root = ray.get(self.tracker.get_checkpoints_dir.remote()) if self.tracker else ckpt_root
#         self.ckpt_root = pathlib.Path(ckpt_root)
#         self.ckpt_root.mkdir(parents=True, exist_ok=True)

#         # — device selection —
#         gpu_ids = ray.get_gpu_ids()
#         self.device = (torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu"))

#         self.model, self.tokenizer = build_peft_model(model_name, self.device, lora_cfg, initial_lora_path)
#         self.model.to(torch.bfloat16)

#         # if num_learners > 1: self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[0], output_device=0, find_unused_parameters=False) # Wrap in DDP (assumes learners launched with 1 GPU each)
        
#         # self.model.config.use_cache = False
#         # if self.gradient_checkpointing:
#         #     self.model.gradient_checkpointing_enable() # gradient checkpointing
        
#         # if self.activation_checkpointing: # TODO fix the qwen3 hardcoding part. Can prob get class from model 
#         # check_fn = make_checkpointing_filter(percentage=1.0, block_class=transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer)
#         # apply_activation_checkpointing(self.model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
    
#         # algorithm and optimizer
#         self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
#         self.accelerator = Accelerator()
#         self.model, self.optimizer = accelerate.prepare(self.model, self.optimizer)


#         self.algorithm.initialize(model=self.model, tokenizer=self.tokenizer, device=self.device, accelerator=self.accelerator)

#         # training counters
#         self._step = 0
#         self._samples_seen = 0


#     def train(self, iterations: int):
#         print("[Learner] starting training loop …")
#         mini_bs = self.batch_size // self.grad_accum

#         while self._step < iterations:
#             while (ray.get(self.step_buffer.size.remote()) < self.batch_size * self.batch_delay_buffer): time.sleep(0.2)

#             batch: List = ray.get(self.step_buffer.get_batch.remote(self.batch_size))
#             assert len(batch) == self.batch_size
#             self._samples_seen += len(batch)
#             self.optimizer.zero_grad(set_to_none=True)

#             metrics_acc: Dict[str, float] = {}
#             for i in range(self.grad_accum):
#                 sub = batch[i * mini_bs : (i + 1) * mini_bs]
#                 update_metrics = self.algorithm.update(sub, scaling=self.grad_accum)
#                 for k, v in update_metrics.items():
#                     metrics_acc[k] = metrics_acc.get(k, 0.0) + v

#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
#             self.optimizer.step()
#             self._step += 1

#             # Logging
#             if self.tracker is not None:
#                 log = {f"{k}": v / self.grad_accum for k, v in metrics_acc.items()}
#                 log.update({
#                     "step": self._step,  "samples_seen": self._samples_seen,  "lr": self.optimizer.param_groups[0]["lr"], 
#                     "grad_norm": sum(p.grad.data.norm(2).item()**2 for p in self.model.parameters() if p.grad is not None) ** 0.5
#                 })
#                 self.tracker.log_learner.remote(log)
#             else:
#                 if self._step % 10 == 0: print(f"[Learner] step {self._step:>5} | loss={metrics_acc.get('loss', 0)/self.grad_accum:.4f}")

#             # save & register checkpoint every step
#             if self._step % self.save_every == 0:
#                 self._save_checkpoint()
#                 if self.model_pool and self._last_ckpt:
#                     self.model_pool.add_checkpoint.remote(str(self._last_ckpt), self._step)
#                     print(f"[Learner] ↪registered → {self._last_ckpt}")
#                     self.model_pool.snapshot.remote(self._step)

#         print("[Learner] training finished.")
#         return {"final_step":self._step, "samples":self._samples}

#     def _save_checkpoint(self):
#         ckpt_dir = self.ckpt_root / f"iteration-{self._step}"
#         ckpt_dir.mkdir(parents=True, exist_ok=True)
#         model = self.model.module if hasattr(self.model,'module') else self.model
#         model.save_pretrained(ckpt_dir, save_adapter=True)
#         self._last_ckpt = ckpt_dir
#         print(f"[Learner] saved → {ckpt_dir}")
