import time 
from typing import List, Tuple, Dict, Optional

import ray

from transformers import AutoModelForCausalLM, AutoTokenizer

# local imports
from unstable.trajectory_buffer import StepBuffer
from unstable.tracker import Tracker
from unstable.collector import Collector
from unstable.algorithms import BaseAlgo
from unstable.utils.lora import load_lora_state



@ray.remote
class Learner:
    def __init__(
        self,
        model_name: str,
        iterations: int,
        num_learners: int = 1,
        learning_rate: float = 1e-4,
        gradient_clip: float = 1.0,
        batch_size: int = 384,
        gradient_accumulation_steps: int = 384,
        batch_delay_buffer: float = 1.5, # an extra collection buffer to make sure we are not biasing the first training step too much
        step_buffer: StepBuffer,
        tracker: Tracker,
        collector: Collector,
        algorithm_class: BaseAlgo,
        lora_cfg: Optional[Dict[str, Any]] = None,
        initial_lora_path: Optional[str] = None
    ):
        self.model_name = model_name 
        self.iterations = iterations
        self.num_learners = num_learners

        self.step_buffer = step_buffer 
        self.tracker = tracker
        self.collector = collector 

        # learning params
        self.gradient_clip = gradient_clip
        self.batch_size = batch_size
        self.batch_delay_buffer = batch_delay_buffer
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self._build_model(lora_cfg=lora_cfg, initial_lora_path=initial_lora_path) # build the model
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate) # optimizer over only the adapters
        self.algorithm = algorithm_class(model=self.model, tokenizer=self.tokenizer, device=self.device) # initialize the algorithm

        # determine mini-batch size
        ctx = get_context()
        rank = ctx.get_world_rank()
        world_size = ctx.get_world_size()
        self.gpu_batch_size = batch_size // world_size


    def _build_model(self, lora_cfg: Dict[str, Any], initial_lora_path: Optional[str] = None):
        local_gpu = rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{local_gpu}")

        # load base + LoRA
        base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
        peft_model = build_lora_model(model=base, lora_cfg=lora_cfg).to(self.device) 
        if initial_lora_path: load_lora_state(peft_model, initial_lora_path) # load weights if available
        self.model = torch.nn.parallel.DistributedDataParallel(peft_model, device_ids=[local_gpu], output_device=local_gpu, find_unused_parameters=False)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)


    def train(self):
        for iteration in range(self.iterations):
            # wait until data is available
            while ray.get(self.buffer.size.remote()) < self.batch_size * self.batch_delay_buffer: time.sleep(0.2)

            # pull batch
            batch = ray.get(self.buffer.get_batch.remote(self.gpu_batch_size))
            self.optimizer.zero_grad(set_to_none=True)

            metrics = {}
            mini_batch_size = len(batch)//self.gradient_accumulation_steps
            for i in range(self.gradient_accumulation_steps):
                start, end = i*mini_batch_size, (i+1)*mini_batch_size # TODO adjust the scaling factor
                update_info = self.algorithm.update(steps=mini_batch, scaling=self.gradient_accumulation_steps)
                for k in update_info:
                    metrics[k] = metrics.get(k, 0.0) + update_info[k]

            # step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            if rank == 0:
                # pass to wandb tracker for logging
                avg_metrics = {f"learner/{k}": v / self.gradient_accumulation_steps for k, v in metrics.items()}
                avg_metrics["learner/iteration"] = iteration
                avg_metrics["learner/grad_norm"] = sum(p.grad.data.norm(2).item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
                avg_metrics["learner/lr"] = optimizer.param_groups[0]["lr"]
                self.tracker.log_training_step(avg_metrics)

                # store checkpoint TODO
                checkpoint_folder_path = os.path.join(root_checkpoint_dir, f"iteration-{iteration}")
                os.makedirs(checkpoint_folder_path, exist_ok=True)
                peft_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                peft_model.save_pretrained(checkpoint_folder_path)
                ray.get(collector.add_new_lora_paths.remote(checkpoint_folder_path, iteration))
                session.report({"iteration": iteration})

        # let the collector know that we are done
        if rank == 0: ray.get(self.collector.mark_done.remote())


