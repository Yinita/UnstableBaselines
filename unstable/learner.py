import os, time, ray, torch, torch.distributed as dist
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file as safe_load
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig


# local imports
from unstable.buffer import StepBuffer
from unstable.core import BaseAlgo
from unstable.model_pool import ModelPool


for k in ("NCCL_SOCKET_IFNAME", "NCCL_NET"):
    os.environ.pop(k, None)
os.environ.update(
    NCCL_P2P_LEVEL="SYS", NCCL_SHM_DISABLE="0", NCCL_IB_DISABLE="1", NCCL_PLUGIN_DISABLE="1",
    NCCL_DEBUG="INFO", TORCH_NCCL_BLOCKING_WAIT="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
)


def _load_lora_state(peft_model, ckpt_dir: str | Path):
    ckpt_dir = Path(ckpt_dir)
    for fn in ("adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin"):
        p = ckpt_dir / fn
        if p.exists():
            sd = safe_load(str(p)) if p.suffix == ".safetensors" else torch.load(p, map_location="cpu")
            set_peft_model_state_dict(peft_model, sd, adapter_name="default")
            return
    raise FileNotFoundError(f"No adapter_model.* in {ckpt_dir}")

def _build_peft_model(base_name: str, device: torch.device, lcfg: Dict[str,Any], init_path: Optional[str]):
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    for p in base.parameters(): p.requires_grad_(False)

    cfg = LoraConfig(
        r=lcfg["lora_rank"], lora_alpha=lcfg["lora_alpha"], lora_dropout=lcfg["lora_dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=lcfg["target_modules"],
    )
    model = get_peft_model(base, cfg).to(device)
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.bfloat16:   # LoRA params
            p.data = p.data.to(torch.bfloat16)

    if init_path:
        _load_lora_state(model, init_path)

    tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    return model, tok


@ray.remote
class Learner:
    def __init__(
        self,
        rank: int,
        world_size: int,
        model_name: str,
        step_buffer: StepBuffer,
        model_pool: ModelPool,
        algorithm: BaseAlgo,
        batch_size: int  = 384,
        gradient_accum: int  = 32,
        lr: float= 5e-6,
        grad_clip: float= 1.0,
        delay_mult: float= 1.5,
        lora_cfg: Dict[str,Any] | None = None,
        initial_lora: Optional[str] = None,
        ckpt_root: str  = "checkpoints",
        save_every: int  = 1,
        tracker=None,
    ):
        # −− device −−
        torch.cuda.set_device(0)
        self.device = torch.device("cuda")

        # −− create /tmp file for PG bootstrap −−
        init_file = "/tmp/learner_pg_shared"
        if rank == 0 and os.path.exists(init_file):
            os.remove(init_file)

        dist.init_process_group(backend="nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)

        # −− model ➜ FSDP −−
        self.model, self.tokenizer = _build_peft_model(model_name, self.device, lora_cfg or {}, initial_lora)
        self.model = FSDP(self.model, use_orig_params=True)

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.algorithm = algorithm
        self.algorithm.initialize(self.model, self.tokenizer, self.device)

        # bookkeeping
        self.step_buffer = step_buffer
        self.model_pool = model_pool
        self.batch_size = batch_size
        self.grad_accum = gradient_accum
        self.grad_clip = grad_clip
        self.delay_mult = delay_mult
        self.tracker = tracker
        self.save_every = save_every
        self._step = 0; self._samples = 0

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
        mini = self.batch_size // self.grad_accum #// 2
        self.optimizer.zero_grad(set_to_none=True)

        while self._step < iterations:
            # wait until buffer has enough samples
            while ray.get(self.step_buffer.size.remote()) < self.batch_size * self.delay_mult:
                time.sleep(0.2)

            # dist.barrier()
            batch = ray.get(self.step_buffer.get_batch.remote(self.batch_size//2))
            # dist.barrier()

            print(f"[RANK {rank}] received {len(batch)} batch_size.", flush=True)

            # batch = ray.get(self.step_buffer.get_batch.remote(self.batch_size))
            self._samples += len(batch)

            for i in range(self.grad_accum//2):
                sub = batch[i*mini : (i+1)*mini]
                print(f"[RANK {rank}] received {len(sub)} mini batch size. Going from {i*mini} to {(i+1)*mini}.", flush=True)
                self.algorithm.update(sub, scaling=self.grad_accum)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._step += 1

            if self._step % 5 == 0 and dist.get_rank() == 0:
                print(f"[Learner-0] step {self._step}", flush=True)

            if self._step % self.save_every == 0:
                self._save_checkpoint()

            dist.barrier()

        dist.destroy_process_group()
        return f"rank {dist.get_rank()} done"

    def _save_checkpoint(self):
        d = self.ckpt_root / f"iter-{self._step}"
        if dist.get_rank() == 0:
            os.makedirs(d, exist_ok=True)

        # --- everyone reaches the barrier synchronously ----
        dist.barrier(device_ids=[torch.cuda.current_device()])

        if dist.get_rank() == 0: # only rank-0 writes
            # collect only parameters that require_grad ⇒ LoRA weights
            lora_sd = {k: p.detach().cpu() for k, p in (self.model.module if hasattr(self.model, "module") else self.model).named_parameters() if p.requires_grad}
            safetensors.torch.save_file(lora_sd, d / "adapter_model.safetensors")

            # write config
            import json, pathlib
            cfg = next(iter(self.model.peft_config.values()))
            json.dump(cfg.to_dict(), open(d / "adapter_config.json", "w"))

            if self.model_pool:
                self.model_pool.add_checkpoint.remote(str(d), self._step)

        # allow rank-1 to continue once the file is on disk
        dist.barrier(device_ids=[torch.cuda.current_device()])
        print(f"[Learner-0] ckpt → {d}", flush=True)
        torch.cuda.empty_cache()

    # def _save_checkpoint(self):
    #     d = self.ckpt_root / f"iter-{self._step}"
    #     os.makedirs(d, exist_ok=True) if dist.get_rank() == 0 else None

    #     # everybody enters the FULL_STATE_DICT context
    #     full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    #     with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_cfg):
    #         full_sd = self.model.state_dict() # ← collective; all ranks must call

    #     if dist.get_rank() == 0:
    #         base = AutoModelForCausalLM.from_pretrained(
    #             self.tokenizer.name_or_path, torch_dtype=torch.bfloat16)
    #         cfg  = self.model.peft_config["default"]
    #         peft_model = get_peft_model(base, cfg)

    #         set_peft_model_state_dict(peft_model, full_sd, adapter_name="default")
    #         peft_model.save_pretrained(d, save_adapter=True)

    #         if self.model_pool:
    #             self.model_pool.add_checkpoint.remote(str(d), self._step)
    #     dist.barrier(device_ids=[torch.cuda.current_device()])  # everyone syncs out

