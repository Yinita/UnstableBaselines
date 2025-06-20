import os, json, time, ray, torch, torch.distributed as dist
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

# from safetensors.torch import safetensors
from torch.distributed.fsdp.wrap import enable_wrap, wrap, transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, StateDictType, FullStateDictConfig, BackwardPrefetch, ShardingStrategy
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
# local imports
from unstable.buffer import StepBuffer
from unstable.core import BaseAlgo
from unstable.model_pool import ModelPool
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt, _json_safe



@ray.remote
class MultiGPULearner:
    """
    Distributed learner (FSDP or DDP) that shares a Ray worker **per rank**.
    Only rank-0 writes checkpoints and registers them; other ranks synchronise via barriers around I/O.
    """
    def __init__(
        self, rank: int, world_size: int, use_fsdp: bool, model_name: str, step_buffer: StepBuffer, model_pool: ModelPool,
        algorithm: BaseAlgo, batch_size: int=384, gradient_accum: int=32, lr: float=5e-6, grad_clip: float=1.0, delay_mult: float=1.5,
        lora_cfg: Dict[str,Any]|None=None, initial_lora: Optional[str]=None, ckpt_root: str="checkpoints", save_every: int=1, tracker=None,
        verbose: bool = True, activation_checkpointing: bool = True, gradient_checkpointing: bool = True, mixed_precision_training: bool = True
    ):
        # TODO docstring
        self.step_buffer = step_buffer
        self.model_pool = model_pool
        self.batch_size = batch_size
        self.grad_accum = gradient_accum
        self.grad_clip = grad_clip
        self.delay_mult = delay_mult
        self.tracker = tracker
        self.save_every = save_every
        self.verbose = verbose
        self._step = 0; self._samples = 0

        torch.cuda.set_device(0)
        self.device = torch.device("cuda")

        init_file = "/tmp/learner_pg_shared_4" # TODO fix this
        if rank == 0 and os.path.exists(init_file): os.remove(init_file)
        dist.init_process_group(backend="nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)

        # model -> FSDP 
        self.model, self.tokenizer = build_peft_model(model_name, self.device, lora_cfg or {}, initial_lora)

        if use_fsdp:
            mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16) if mixed_precision_training else None
            self.model = FSDP(self.model, auto_wrap_policy=transformer_auto_wrap_policy({Qwen3DecoderLayer}), sharding_strategy=ShardingStrategy.FULL_SHARD, mixed_precision=mp, device_id=self.device, use_orig_params=True)
        else:
            self.model = DDP(self.model, device_ids=[rank], broadcast_buffers=False)

        if gradient_checkpointing:   self.model.gradient_checkpointing_enable()
        if activation_checkpointing: enable_full_activation_ckpt(self.model)


        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.algorithm = algorithm
        self.algorithm.initialize(self.model, self.tokenizer, self.device)

        self.ckpt_root = Path(ray.get(tracker.get_checkpoints_dir.remote()) if tracker else ckpt_root)
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

    def ready(self):
        print(f"[Learner-{os.getenv('RANK', '?')}] ready", flush=True)
        return True

    def synced(self):
        dist.barrier(device_ids=[torch.cuda.current_device()])
        return True

    # training loop
    def train(self, iterations: int):
        world = dist.get_world_size()
        rank  = dist.get_rank()
        mini = self.batch_size // self.grad_accum
        self.optimizer.zero_grad(set_to_none=True)

        while self._step < iterations:
            # wait until buffer has enough samples
            while ray.get(self.step_buffer.size.remote()) < self.batch_size * self.delay_mult: time.sleep(0.2)

            batch = ray.get(self.step_buffer.get_batch.remote(self.batch_size//2))
            if self.verbose: print(f"[RANK {rank}] received {len(batch)} batch_size.", flush=True)
            self._samples += len(batch)

            for i in range(self.grad_accum//2):
                sub = batch[i*mini : (i+1)*mini]
                print(f"[RANK {rank}] received {len(sub)} mini batch size. Going from {i*mini} to {(i+1)*mini}.", flush=True)
                self.algorithm.update(sub, scaling=self.grad_accum)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._step += 1
            if self._step % 5 == 0 and dist.get_rank() == 0: print(f"[Learner-0] step {self._step}", flush=True)
            if self._step % self.save_every == 0: self._save_checkpoint()
            dist.barrier()

        dist.destroy_process_group()
        return f"rank {dist.get_rank()} done"

    def _save_checkpoint(self):
        d = self.ckpt_root / f"iter-{self._step}"
        if dist.get_rank() == 0: os.makedirs(d, exist_ok=True)
        dist.barrier(device_ids=[torch.cuda.current_device()]) # ALL RANKS join before / after writing
        if dist.get_rank() == 0:
            # 1. Collect only parameters with requires_grad == True  (i.e. LoRA)
            lora_sd_raw = {k: p.detach().cpu() for k, p in (self.model.module if hasattr(self.model, "module") else self.model).named_parameters() if p.requires_grad}
            # 2. Normalise key names for vLLM/PEFT:
            lora_sd_fixed = {}
            for k, v in lora_sd_raw.items():
                k = k.replace("base_model.model.", "", 1).replace(".default", "").replace("._checkpoint_wrapped_module", "")
                lora_sd_fixed[k] = v
            # 3. Write adapter weights and config
            safetensors.save_file(lora_sd_fixed, d / "adapter_model.safetensors")
            cfg = next(iter(self.model.peft_config.values()))
            with open(d / "adapter_config.json", "w") as f:
                json.dump(cfg.to_dict(), f, default=_json_safe, indent=2)
            # 4. Register with model-pool
            if self.model_pool: self.model_pool.add_checkpoint.remote(str(d), self._step)
        # ensure every rank waits until files are on disk
        dist.barrier(device_ids=[torch.cuda.current_device()])
        if dist.get_rank() == 0 and self.verbose:
            print(f"[Learner-0] ckpt -> {d}", flush=True)
        torch.cuda.empty_cache()

