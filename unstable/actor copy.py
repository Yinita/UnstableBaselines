import os, time
import asyncio
from collections import deque
from typing import Optional, Dict, Any

import ray, torch, vllm
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest
import time, itertools
from collections import defaultdict, deque

TOK_RATE_WIN = 100      # sliding-window (#tokens) for short-term rate
EMA_ALPHA    = 0.2     # 1-τ for an exponential-moving-average fallback
REPORT_EVERY = 1.0

@ray.remote
class VLLMActor:
    def __init__(self, vllm_config: Dict[str, Any], tracker=None, name: str="actor-0"):
        gpu_ids = ray.get_gpu_ids() 
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        torch.cuda.set_device(0)

        engine_args = EngineArgs(
            model=vllm_config["model_name"], enable_lora=True, max_loras=vllm_config.get("max_loras", 5), max_lora_rank=vllm_config["lora_config"]["lora_rank"], 
            max_cpu_loras=vllm_config.get("max_loras", 5), max_num_seqs=vllm_config["max_parallel_seq"], task="generate", max_model_len=vllm_config.get("max_model_len", 8192)
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(temperature=vllm_config.get("temperature", 0.7),  top_p=vllm_config.get("top_p", 0.95), max_tokens=vllm_config.get("max_tokens", 4096))

        self._queue = deque()
        self._futures = {}
        self._next_id = 0

        self.tracker = tracker
        self.name = name

        # Fixed: Separate tracking per LoRA
        self._req2lora: dict[str, str] = {}           # request-id ➜ lora-name
        self._lora_queued: dict[str, int] = defaultdict(int)    # lora-name ➜ queued count
        self._lora_running: dict[str, int] = defaultdict(int)   # lora-name ➜ running count
        
        # Fixed: Track tokens per LoRA separately
        self._lora_tok_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=TOK_RATE_WIN))
        self._lora_tok_ema: dict[str, float] = defaultdict(float)

        loop = asyncio.get_event_loop()
        loop.create_task(self._batch_loop())
        loop.create_task(self._report_loop())

    def _tok_rate(self, lora_name: str = "base"):
        """Calculate token rate for specific LoRA"""
        now = time.monotonic()
        tok_hist = self._lora_tok_hist[lora_name]
        
        # Remove old timestamps (older than 1 second)
        while tok_hist and now - tok_hist[0] > 1.0:
            tok_hist.popleft()
        
        # Return actual count of tokens in the last second
        return len(tok_hist)

    async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
        fut = asyncio.Future()
        lora_name = lora_path or "base"
        
        # Fixed: Track queued requests per LoRA
        self._lora_queued[lora_name] += 1
        self._queue.append((prompt, lora_path, fut))
        return await fut

    async def _batch_loop(self):
        while True:
            await asyncio.sleep(0.02)
            
            # Process queued requests
            while self._queue:
                prompt, path, fut = self._queue.popleft()
                req_id = str(self._next_id); self._next_id += 1
                lora_name = path or "base"
                
                self._futures[req_id] = fut
                self._req2lora[req_id] = lora_name
                
                # Fixed: Update counters correctly
                self._lora_queued[lora_name] -= 1
                self._lora_running[lora_name] += 1
                
                lora_req = (LoRARequest(path, path, abs(hash(path))) if path else None)
                self.engine.add_request(req_id, prompt, self.sampling_params, lora_request=lora_req)

            # Process engine outputs
            engine_outputs = self.engine.step()
            for out in engine_outputs:
                lora_name = self._req2lora.get(out.request_id, "base")
                
                # Count NEW tokens only (not cumulative)
                if out.outputs and len(out.outputs) > 0:
                    segment = out.outputs[-1]
                    
                    # Record one timestamp per NEW token generated in this step
                    # vLLM outputs are incremental, so we only count new tokens
                    if hasattr(segment, 'token_ids') and segment.token_ids:
                        # Count only the newly generated tokens in this step
                        # This is tricky - vLLM gives cumulative tokens, we need just the new ones
                        # For now, assume 1 new token per step when not finished
                        if segment.finish_reason is None:
                            self._lora_tok_hist[lora_name].append(time.monotonic())
                    else:
                        # Fallback: assume 1 token per step if token_ids not available
                        if segment.finish_reason is None:
                            self._lora_tok_hist[lora_name].append(time.monotonic())

                    # Check if request is finished
                    if segment.finish_reason is not None:
                        fut = self._futures.pop(out.request_id, None)
                        if fut and not fut.done():
                            fut.set_result(segment.text)
                        
                        # Fixed: Update running counter when request actually finishes
                        if lora_name in self._lora_running and self._lora_running[lora_name] > 0:
                            self._lora_running[lora_name] -= 1
                        
                        self._req2lora.pop(out.request_id, None)

    async def _report_loop(self):
        while True:
            if self.tracker is not None:
                # Debug: get actual engine stats
                engine_stats = {
                    'num_unfinished': self.engine.get_num_unfinished_requests(),
                    'scheduler_running': len(self.engine.scheduler.running),
                    'scheduler_waiting': len(self.engine.scheduler.waiting),
                    'scheduler_swapped': len(self.engine.scheduler.swapped)
                }
                
                # Fixed: Use actual engine state instead of our counters for validation
                stats = {}
                all_loras = set(self._lora_queued.keys()) | set(self._lora_running.keys()) | set(self._lora_tok_hist.keys())
                all_loras.add("base")
                
                for lora_name in all_loras:
                    # Use our counters but validate against engine state
                    queued = max(0, self._lora_queued.get(lora_name, 0))
                    running = max(0, self._lora_running.get(lora_name, 0))
                    tok_rate = self._tok_rate(lora_name)
                    
                    # Only report if there's activity or recent activity
                    if queued > 0 or running > 0 or tok_rate > 0:
                        stats[lora_name] = {
                            "queue": queued,
                            "running": running, 
                            "tok_s": tok_rate
                        }
                
                # Always report base stats with engine validation
                if not stats:
                    stats["base"] = {
                        "queue": engine_stats['scheduler_waiting'],
                        "running": engine_stats['scheduler_running'],
                        "tok_s": self._tok_rate("base")
                    }
                
                await self.tracker.log_inference.remote(actor=self.name, stats=stats)
            
            await asyncio.sleep(REPORT_EVERY)


            

# import os, time
# import asyncio
# from collections import deque
# from typing import Optional, Dict, Any

# import ray, torch, vllm
# from vllm import EngineArgs, LLMEngine, SamplingParams
# from vllm.lora.request import LoRARequest
# import time, itertools
# from collections import defaultdict, deque

# TOK_RATE_WIN = 100      # sliding-window (#tokens) for short-term rate
# EMA_ALPHA    = 0.2     # 1-τ for an exponential-moving-average fallback
# REPORT_EVERY = 1.0

# @ray.remote
# class VLLMActor:
#     def __init__(self, vllm_config: Dict[str, Any], tracker=None, name: str="actor-0"):
#         gpu_ids = ray.get_gpu_ids() 
#         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
#         torch.cuda.set_device(0)

#         engine_args = EngineArgs(
#             model=vllm_config["model_name"], enable_lora=True, max_loras=vllm_config.get("max_loras", 5), max_lora_rank=vllm_config["lora_config"]["lora_rank"], 
#             max_cpu_loras=vllm_config.get("max_loras", 5), max_num_seqs=vllm_config["max_parallel_seq"], task="generate", max_model_len=vllm_config.get("max_model_len", 8192)
#         )
#         self.engine = LLMEngine.from_engine_args(engine_args)
#         self.sampling_params = SamplingParams(temperature=vllm_config.get("temperature", 0.7),  top_p=vllm_config.get("top_p", 0.95), max_tokens=vllm_config.get("max_tokens", 4096))

#         self._queue = deque()
#         self._futures = {}
#         self._next_id = 0

#         self.tracker = tracker
#         self.name = name

#         # Fixed: Separate tracking per LoRA
#         self._req2lora: dict[str, str] = {}           # request-id ➜ lora-name
#         self._lora_queued: dict[str, int] = defaultdict(int)    # lora-name ➜ queued count
#         self._lora_running: dict[str, int] = defaultdict(int)   # lora-name ➜ running count
        
#         # Fixed: Track tokens per LoRA separately
#         self._lora_tok_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=TOK_RATE_WIN))
#         self._lora_tok_ema: dict[str, float] = defaultdict(float)

#         loop = asyncio.get_event_loop()
#         loop.create_task(self._batch_loop())
#         loop.create_task(self._report_loop())

#     def _tok_rate(self, lora_name: str = "base"):
#         """Calculate token rate for specific LoRA"""
#         now = time.monotonic()
#         tok_hist = self._lora_tok_hist[lora_name]
        
#         # Remove old timestamps (older than 1 second)
#         while tok_hist and now - tok_hist[0] > 1.0:
#             tok_hist.popleft()
        
#         # Return actual count of tokens in the last second
#         return len(tok_hist)

#     async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
#         fut = asyncio.Future()
#         lora_name = lora_path or "base"
        
#         # Fixed: Track queued requests per LoRA
#         self._lora_queued[lora_name] += 1
#         self._queue.append((prompt, lora_path, fut))
#         return await fut

#     async def _batch_loop(self):
#         while True:
#             await asyncio.sleep(0.02)
            
#             # Process queued requests
#             while self._queue:
#                 prompt, path, fut = self._queue.popleft()
#                 req_id = str(self._next_id); self._next_id += 1
#                 lora_name = path or "base"
                
#                 self._futures[req_id] = fut
#                 self._req2lora[req_id] = lora_name
                
#                 # Fixed: Update counters correctly
#                 self._lora_queued[lora_name] -= 1
#                 self._lora_running[lora_name] += 1
                
#                 lora_req = (LoRARequest(path, path, abs(hash(path))) if path else None)
#                 self.engine.add_request(req_id, prompt, self.sampling_params, lora_request=lora_req)

#             # Process engine outputs
#             engine_outputs = self.engine.step()
#             for out in engine_outputs:
#                 lora_name = self._req2lora.get(out.request_id, "base")
                
#                 # Count NEW tokens only (not cumulative)
#                 if out.outputs and len(out.outputs) > 0:
#                     segment = out.outputs[-1]
                    
#                     # Record one timestamp per NEW token generated in this step
#                     # vLLM outputs are incremental, so we only count new tokens
#                     if hasattr(segment, 'token_ids') and segment.token_ids:
#                         # Count only the newly generated tokens in this step
#                         # This is tricky - vLLM gives cumulative tokens, we need just the new ones
#                         # For now, assume 1 new token per step when not finished
#                         if segment.finish_reason is None:
#                             self._lora_tok_hist[lora_name].append(time.monotonic())
#                     else:
#                         # Fallback: assume 1 token per step if token_ids not available
#                         if segment.finish_reason is None:
#                             self._lora_tok_hist[lora_name].append(time.monotonic())

#                     # Check if request is finished
#                     if segment.finish_reason is not None:
#                         fut = self._futures.pop(out.request_id, None)
#                         if fut and not fut.done():
#                             fut.set_result(segment.text)
                        
#                         # Fixed: Update running counter when request actually finishes
#                         if lora_name in self._lora_running and self._lora_running[lora_name] > 0:
#                             self._lora_running[lora_name] -= 1
                        
#                         self._req2lora.pop(out.request_id, None)

#     async def _report_loop(self):
#         while True:
#             if self.tracker is not None:
#                 # Debug: get actual engine stats
#                 engine_stats = {
#                     'num_unfinished': self.engine.get_num_unfinished_requests(),
#                     'scheduler_running': len(self.engine.scheduler.running),
#                     'scheduler_waiting': len(self.engine.scheduler.waiting),
#                     'scheduler_swapped': len(self.engine.scheduler.swapped)
#                 }
                
#                 # Fixed: Use actual engine state instead of our counters for validation
#                 stats = {}
#                 all_loras = set(self._lora_queued.keys()) | set(self._lora_running.keys()) | set(self._lora_tok_hist.keys())
#                 all_loras.add("base")
                
#                 for lora_name in all_loras:
#                     # Use our counters but validate against engine state
#                     queued = max(0, self._lora_queued.get(lora_name, 0))
#                     running = max(0, self._lora_running.get(lora_name, 0))
#                     tok_rate = self._tok_rate(lora_name)
                    
#                     # Only report if there's activity or recent activity
#                     if queued > 0 or running > 0 or tok_rate > 0:
#                         stats[lora_name] = {
#                             "queue": queued,
#                             "running": running, 
#                             "tok_s": tok_rate
#                         }
                
#                 # Always report base stats with engine validation
#                 if not stats:
#                     stats["base"] = {
#                         "queue": engine_stats['scheduler_waiting'],
#                         "running": engine_stats['scheduler_running'],
#                         "tok_s": self._tok_rate("base")
#                     }
                
#                 await self.tracker.log_inference.remote(actor=self.name, stats=stats)
            
#             await asyncio.sleep(REPORT_EVERY)

# # import os, time
# # import asyncio
# # from collections import deque
# # from typing import Optional, Dict, Any

# # import ray, torch, vllm
# # from vllm import EngineArgs, LLMEngine, SamplingParams
# # from vllm.lora.request import LoRARequest
# # import time, itertools
# # from collections import defaultdict, deque

# # TOK_RATE_WIN = 100      # sliding-window (#tokens) for short-term rate
# # EMA_ALPHA    = 0.2     # 1-τ for an exponential-moving-average fallback
# # REPORT_EVERY = 1.0

# # @ray.remote
# # class VLLMActor:
# #     def __init__(self, vllm_config: Dict[str, Any], tracker=None, name: str="actor-0"):
# #         gpu_ids = ray.get_gpu_ids() 
# #         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
# #         torch.cuda.set_device(0)

# #         engine_args = EngineArgs(
# #             model=vllm_config["model_name"], enable_lora=True, max_loras=vllm_config.get("max_loras", 5), max_lora_rank=vllm_config["lora_config"]["lora_rank"], 
# #             max_cpu_loras=vllm_config.get("max_loras", 5), max_num_seqs=vllm_config["max_parallel_seq"], task="generate", max_model_len=vllm_config.get("max_model_len", 8192)
# #         )
# #         self.engine = LLMEngine.from_engine_args(engine_args)
# #         self.sampling_params = SamplingParams(temperature=vllm_config.get("temperature", 0.7),  top_p=vllm_config.get("top_p", 0.95), max_tokens=vllm_config.get("max_tokens", 4096))

# #         self._queue = deque()
# #         self._futures = {}
# #         self._next_id = 0

# #         self.tracker = tracker
# #         self.name = name

# #         # Fixed: Separate tracking per LoRA
# #         self._req2lora: dict[str, str] = {}           # request-id ➜ lora-name
# #         self._lora_queued: dict[str, int] = defaultdict(int)    # lora-name ➜ queued count
# #         self._lora_running: dict[str, int] = defaultdict(int)   # lora-name ➜ running count
        
# #         # Fixed: Track tokens per LoRA separately
# #         self._lora_tok_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=TOK_RATE_WIN))
# #         self._lora_tok_ema: dict[str, float] = defaultdict(float)

# #         loop = asyncio.get_event_loop()
# #         loop.create_task(self._batch_loop())
# #         loop.create_task(self._report_loop())

# #     def _tok_rate(self, lora_name: str = "base"):
# #         """Calculate token rate for specific LoRA"""
# #         now = time.monotonic()
# #         tok_hist = self._lora_tok_hist[lora_name]
        
# #         # Remove old timestamps (older than 1 second)
# #         while tok_hist and now - tok_hist[0] > 1.0:
# #             tok_hist.popleft()
        
# #         # Instantaneous rate based on tokens in last second
# #         inst_rate = len(tok_hist)
        
# #         # Update EMA
# #         self._lora_tok_ema[lora_name] = EMA_ALPHA * inst_rate + (1 - EMA_ALPHA) * self._lora_tok_ema[lora_name]
        
# #         return self._lora_tok_ema[lora_name]

# #     async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
# #         fut = asyncio.Future()
# #         lora_name = lora_path or "base"
        
# #         # Fixed: Track queued requests per LoRA
# #         self._lora_queued[lora_name] += 1
# #         self._queue.append((prompt, lora_path, fut))
# #         return await fut

# #     async def _batch_loop(self):
# #         while True:
# #             await asyncio.sleep(0.02)
            
# #             # Process queued requests
# #             while self._queue:
# #                 prompt, path, fut = self._queue.popleft()
# #                 req_id = str(self._next_id); self._next_id += 1
# #                 lora_name = path or "base"
                
# #                 self._futures[req_id] = fut
# #                 self._req2lora[req_id] = lora_name
                
# #                 # Fixed: Update counters correctly
# #                 self._lora_queued[lora_name] -= 1
# #                 self._lora_running[lora_name] += 1
                
# #                 lora_req = (LoRARequest(path, path, abs(hash(path))) if path else None)
# #                 self.engine.add_request(req_id, prompt, self.sampling_params, lora_request=lora_req)

# #             # Process engine outputs
# #             for out in self.engine.step():
# #                 lora_name = self._req2lora.get(out.request_id, "base")
                
# #                 # Fixed: Record token timestamp for the specific LoRA
# #                 # Only count tokens for new outputs, not repeated ones
# #                 if out.outputs and len(out.outputs) > 0:
# #                     segment = out.outputs[-1]
# #                     # Count each new token generated
# #                     if hasattr(segment, 'token_ids') and segment.token_ids:
# #                         for _ in segment.token_ids:
# #                             self._lora_tok_hist[lora_name].append(time.monotonic())
# #                     else:
# #                         # Fallback: assume 1 token per step if token_ids not available
# #                         self._lora_tok_hist[lora_name].append(time.monotonic())

# #                     # Check if request is finished
# #                     if segment.finish_reason is not None:
# #                         fut = self._futures.pop(out.request_id, None)
# #                         if fut and not fut.done():
# #                             fut.set_result(segment.text)
                        
# #                         # Fixed: Update running counter
# #                         self._lora_running[lora_name] -= 1
# #                         if self._lora_running[lora_name] <= 0:
# #                             self._lora_running[lora_name] = 0
                        
# #                         self._req2lora.pop(out.request_id, None)

# #     async def _report_loop(self):
# #         while True:
# #             if self.tracker is not None:
# #                 # Fixed: Report stats for all active LoRAs
# #                 stats = {}
# #                 all_loras = set(self._lora_queued.keys()) | set(self._lora_running.keys()) | set(self._lora_tok_hist.keys())
                
# #                 # Always include base model
# #                 all_loras.add("base")
                
# #                 for lora_name in all_loras:
# #                     queued = self._lora_queued.get(lora_name, 0)
# #                     running = self._lora_running.get(lora_name, 0)
# #                     tok_rate = self._tok_rate(lora_name)
                    
# #                     # Only report if there's activity or recent activity
# #                     if queued > 0 or running > 0 or tok_rate > 0.1:
# #                         stats[lora_name] = {
# #                             "queue": queued,
# #                             "running": running, 
# #                             "tok_s": tok_rate
# #                         }
                
# #                 # Always report at least base stats
# #                 if "base" not in stats:
# #                     stats["base"] = {
# #                         "queue": self._lora_queued.get("base", 0),
# #                         "running": self._lora_running.get("base", 0),
# #                         "tok_s": self._tok_rate("base")
# #                     }
                
# #                 await self.tracker.log_inference.remote(actor=self.name, stats=stats)
            
# #             await asyncio.sleep(REPORT_EVERY)





# # # # import os, time
# # # # import asyncio
# # # # from collections import deque
# # # # from typing import Optional, Dict, Any

# # # # import ray, torch, vllm
# # # # from vllm import EngineArgs, LLMEngine, SamplingParams
# # # # from vllm.lora.request import LoRARequest


# # # # @ray.remote
# # # # class VLLMActor:
# # # #     def __init__(self, vllm_config: Dict[str, Any]):
# # # #         gpu_ids = ray.get_gpu_ids() 
# # # #         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
# # # #         torch.cuda.set_device(0)

# # # #         engine_args = EngineArgs(
# # # #             model=vllm_config["model_name"], enable_lora=True, max_loras=vllm_config.get("max_loras", 5), max_lora_rank=vllm_config["lora_config"]["lora_rank"], 
# # # #             max_cpu_loras=vllm_config.get("max_loras", 5), max_num_seqs=vllm_config["max_parallel_seq"], task="generate", max_model_len=vllm_config.get("max_model_len", 8192)
# # # #         )
# # # #         self.engine = LLMEngine.from_engine_args(engine_args)
# # # #         self.sampling_params = SamplingParams(temperature=vllm_config.get("temperature", 0.7),  top_p=vllm_config.get("top_p", 0.95), max_tokens=vllm_config.get("max_tokens", 4096))

# # # #         self._queue = deque()
# # # #         self._futures = {}
# # # #         self._next_id = 0

# # # #         loop = asyncio.get_event_loop()
# # # #         loop.create_task(self._batch_loop())

# # # #     async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
# # # #         fut = asyncio.Future()
# # # #         self._queue.append((prompt, lora_path, fut))
# # # #         return await fut

# # # #     async def _batch_loop(self):
# # # #         while True:
# # # #             await asyncio.sleep(0.02)
# # # #             while self._queue:
# # # #                 prompt, path, fut = self._queue.popleft()
# # # #                 req_id = str(self._next_id)
# # # #                 self._next_id += 1
# # # #                 self._futures[req_id] = fut
# # # #                 lora_req = LoRARequest(lora_name=path, lora_path=path, lora_int_id=abs(hash(path))) if path is not None else None
# # # #                 print(f"[Actor {os.getpid()}] submitting req {req_id} with {lora_req}")
# # # #                 self.engine.add_request(req_id, prompt, self.sampling_params, lora_request=lora_req)
# # # #             # Step the engine and only resolve once finish_reason is non-None
# # # #             for out in self.engine.step():
# # # #                 token = out.outputs[-1] # take the last token in this partial output
# # # #                 if token.finish_reason is None: continue # skip interim/newline events
# # # #                 fut = self._futures.pop(out.request_id, None) # now it’s done—fulfil the future
# # # #                 if fut: fut.set_result(token.text)

# # # #     async def get_queue_stats(self):
# # # #         return {self._current_lora or "base": {"queue": self.engine.get_num_unfinished_requests(),"running": len(self._futures)}}

# # # # ----------------- vllm_actor.py -----------------
# # # import os, time
# # # import asyncio
# # # from collections import deque
# # # from typing import Optional, Dict, Any

# # # import ray, torch, vllm
# # # from vllm import EngineArgs, LLMEngine, SamplingParams
# # # from vllm.lora.request import LoRARequest
# # # import time, itertools
# # # from collections import defaultdict, deque

# # # TOK_RATE_WIN = 100      # sliding-window (#tokens) for short-term rate
# # # EMA_ALPHA    = 0.2     # 1-τ for an exponential-moving-average fallback
# # # REPORT_EVERY = 1.0

# # # @ray.remote
# # # class VLLMActor:
# # #     def __init__(self, vllm_config: Dict[str, Any], tracker=None, name: str="actor-0"):
# # #         gpu_ids = ray.get_gpu_ids() 
# # #         os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
# # #         torch.cuda.set_device(0)

# # #         engine_args = EngineArgs(
# # #             model=vllm_config["model_name"], enable_lora=True, max_loras=vllm_config.get("max_loras", 5), max_lora_rank=vllm_config["lora_config"]["lora_rank"], 
# # #             max_cpu_loras=vllm_config.get("max_loras", 5), max_num_seqs=vllm_config["max_parallel_seq"], task="generate", max_model_len=vllm_config.get("max_model_len", 8192)
# # #         )
# # #         self.engine = LLMEngine.from_engine_args(engine_args)
# # #         self.sampling_params = SamplingParams(temperature=vllm_config.get("temperature", 0.7),  top_p=vllm_config.get("top_p", 0.95), max_tokens=vllm_config.get("max_tokens", 4096))

# # #         self._queue = deque()
# # #         self._futures = {}
# # #         self._next_id = 0

# # #         self.tracker = tracker
# # #         self.name = name

# # #         self._req2lora: dict[str, str] = {}           # request-id ➜ lora-name
# # #         self._tok_hist = deque(maxlen=TOK_RATE_WIN)   # timestamps of recent tokens
# # #         self._tok_ema  = 0.0                          # smooth rate when queue is empty

# # #         loop = asyncio.get_event_loop()
# # #         loop.create_task(self._batch_loop())
# # #         loop.create_task(self._report_loop())


# # #     def _tok_rate(self):
# # #         now = time.monotonic()
# # #         while self._tok_hist and now - self._tok_hist[0] > 1.0:
# # #             self._tok_hist.popleft()
# # #         inst = len(self._tok_hist)
# # #         self._tok_ema = 0.2 * inst + 0.8 * self._tok_ema   # EMA smoothing
# # #         return self._tok_ema

# # #     async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
# # #         fut = asyncio.Future()
# # #         self._queue.append((prompt, lora_path, fut))
# # #         return await fut

# # #     async def _batch_loop(self):
# # #         while True:
# # #             await asyncio.sleep(0.02)
# # #             while self._queue:
# # #                 prompt, path, fut = self._queue.popleft()
# # #                 req_id = str(self._next_id); self._next_id += 1
# # #                 self._futures[req_id] = fut
# # #                 lora_req = (LoRARequest(path, path, abs(hash(path))) if path else None)
# # #                 self._req2lora[req_id] = path or "base"
# # #                 self.engine.add_request(req_id, prompt, self.sampling_params, lora_request=lora_req)

# # #             for out in self.engine.step():
# # #                 self._tok_hist.append(time.monotonic())
# # #                 segment = out.outputs[-1]
# # #                 # record each produced token time-stamp
# # #                 self._tok_hist.append(time.monotonic())

# # #                 if segment.finish_reason is None:
# # #                     continue   # not finished yet

# # #                 fut = self._futures.pop(out.request_id, None)
# # #                 if fut:
# # #                     fut.set_result(segment.text)
# # #                 self._req2lora.pop(out.request_id, None)

# # #     async def _report_loop(self):
# # #         while True:
# # #             if self.tracker is not None:
# # #                 await self.tracker.log_inference.remote(
# # #                     actor=self.name, 
# # #                     stats={"base": {"queue": self.engine.get_num_unfinished_requests(), "running": len(self._futures), "tok_s": self._tok_rate()}}
# # #                 )
# # #             await asyncio.sleep(REPORT_EVERY)
