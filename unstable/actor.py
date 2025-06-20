import os, time, asyncio
from collections import defaultdict, deque
from typing import Optional, Dict, Any

import ray, torch
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

import logging, pathlib
from unstable.utils.logging import setup_logger



@ray.remote
class VLLMActor:
    def __init__(self, vllm_config: Dict[str, Any], tracker, name: str):
        # CRITICAL: Add these CUDA environment variables to prevent deadlocks
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable blocking CUDA calls
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Better error handling
        os.environ["NCCL_TIMEOUT"] = "600"  # 10 minute timeout
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Consistent GPU ordering
        
        # Add CUDA memory management to prevent fragmentation deadlocks
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # set up logging
        log_dir = ray.get(tracker.get_log_dir.remote())
        self.logger = setup_logger(f"actor-{name}", log_dir)

        self.gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        
        # CRITICAL: Add explicit CUDA initialization and error checking
        try:
            import torch.cuda
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()  # Clear any existing memory
            self.logger.info(f"CUDA initialized on device {torch.cuda.current_device()}")
        except Exception as e:
            self.logger.error(f"CUDA initialization failed: {e}")
            raise

        engine_args = EngineArgs(
            model = vllm_config["model_name"], 
            enable_lora = True, 
            max_loras = vllm_config.get("max_loras", 5),
            max_lora_rank = vllm_config["lora_config"]["lora_rank"], 
            max_cpu_loras = vllm_config.get("max_loras", 5),
            max_num_seqs = vllm_config["max_parallel_seq"], 
            task = "generate", 
            max_model_len = vllm_config.get("max_model_len", 8192),
            # CRITICAL: Add these VLLM-specific deadlock prevention settings
            disable_custom_all_reduce = True,  # Prevent NCCL deadlocks
            enforce_eager = False,  # Allow lazy execution
            disable_log_stats = True,  # Reduce logging overhead
        )
        
        try:
            self.engine = LLMEngine.from_engine_args(engine_args)
            self.logger.info("VLLM engine initialized successfully")
        except Exception as e:
            self.logger.error(f"VLLM engine initialization failed: {e}")
            raise
            
        self.sampling_params = SamplingParams(
            temperature = vllm_config.get("temperature", 0.7), 
            top_p = vllm_config.get("top_p", 0.95), 
            max_tokens = vllm_config.get("max_tokens", 4096),
        )

        # bookkeeping
        self._queue = deque()
        self._futures = {}
        self._next_id = 0
        self._req2lora = {}
        self._prev_tok_cnt = defaultdict(int)

        self.tracker = tracker
        self.name = name

        self._queued = defaultdict(int)
        self._running = defaultdict(int)
        self._tok_hist = defaultdict(lambda: deque())

        # CRITICAL: Add explicit event loop management
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        self._batch_task = loop.create_task(self._batch_loop())
        self._report_task = loop.create_task(self._report_loop())

        self._lora_ids: Dict[str, int] = {"base": 0}
        self._next_lora_id = 1
        
        # Add health check flag
        self._last_step_time = time.monotonic()

    async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
        fut = asyncio.Future()
        lora = lora_path or "base"
        self._queued[lora] += 1
        self._queue.append((prompt, lora_path, fut))
        return await fut

    async def _batch_loop(self):
        while True:
            try:
                await asyncio.sleep(0.02)
                
                # CRITICAL: Add deadlock detection
                now = time.monotonic()
                if now - self._last_step_time > 30:  # 30 second deadlock detection
                    self.logger.error(f"Potential deadlock detected - no engine steps for {now - self._last_step_time:.1f} seconds")
                    self.logger.error(f"Running requests: {dict(self._running)}")
                    self.logger.error(f"Queue size: {len(self._queue)}")
                
                while self._queue:
                    prompt, path, fut = self._queue.popleft()
                    lora = path or "base"
                    req_id = str(self._next_id); self._next_id += 1

                    self._futures[req_id]  = fut
                    self._req2lora[req_id] = lora
                    self._queued[lora]    -= 1
                    self._running[lora]   += 1

                    if path:
                        if path not in self._lora_ids:
                            self._lora_ids[path] = self._next_lora_id
                            self._next_lora_id += 1
                        lora_req = LoRARequest(path, self._lora_ids[path], path)
                    else:
                        lora_req = None
                        
                    try:
                        self.engine.add_request(req_id, prompt, self.sampling_params, lora_request=lora_req)
                        self.logger.debug(f"Added request {req_id} with lora {lora}")
                    except Exception as e:
                        self.logger.error(f"Failed to add request {req_id}: {e}")
                        # Clean up the failed request
                        self._running[lora] -= 1
                        self._req2lora.pop(req_id, None)
                        fut.set_exception(e)
                        continue

                # CRITICAL: Add timeout and better error handling to engine.step()
                try:
                    step_start = time.monotonic()
                    outs = self.engine.step()
                    step_duration = time.monotonic() - step_start
                    self._last_step_time = time.monotonic()
                    
                    if step_duration > 5.0:  # Log slow steps
                        self.logger.warning(f"Slow engine step: {step_duration:.1f}s")
                        
                except Exception as exc:   
                    self.logger.exception(f"engine.step() failed - running: {dict(self._running)}")
                    # Instead of raising (which kills the actor), try to recover
                    await asyncio.sleep(1.0)  # Brief pause before retry
                    continue

                for out in outs:
                    req_id = out.request_id
                    lora = self._req2lora.get(req_id, "base")
                    segment = out.outputs[-1]

                    tok_ids = getattr(segment, "token_ids", None) or []
                    prev = self._prev_tok_cnt[req_id]
                    new_tok = max(0, len(tok_ids) - prev)
                    self._prev_tok_cnt[req_id] = len(tok_ids)

                    now = time.monotonic()
                    for _ in range(new_tok):
                        self._tok_hist[lora].append(now)

                    if segment.finish_reason is not None:
                        fut = self._futures.pop(req_id, None)
                        if fut and not fut.done():
                            fut.set_result(segment.text)

                        self._running[lora] -= 1
                        self._req2lora.pop(req_id, None)
                        self._prev_tok_cnt.pop(req_id, None)
                        
            except Exception as e:
                self.logger.exception(f"Critical error in batch loop: {e}")
                await asyncio.sleep(1.0)  # Prevent tight error loop

    # Add health check method
    async def get_health_status(self):
        """Health check method to detect if actor is responsive"""
        return {
            "name": self.name,
            "running_requests": dict(self._running),
            "queued_requests": dict(self._queued),
            "last_step_time": self._last_step_time,
            "time_since_last_step": time.monotonic() - self._last_step_time,
            "futures_count": len(self._futures),
            "engine_loaded": hasattr(self, 'engine') and self.engine is not None
        }

    async def _report_loop(self):
        while True:
            if self.tracker is not None:
                try:
                    lora_stats = {}
                    for l in {*self._queued, *self._running, *self._tok_hist}:
                        tok_s = self._tok_rate(l)
                        lora_stats[l] = {"queued": self._queued[l], "running": self._running[l], "tok_s": tok_s}
                        self.logger.info(f"Lora: [{l}]\n\t- Queued: {self._queued[l]}\n\t- Running: {self._running[l]}\n\t- Token/second: {tok_s}")
                    await self.tracker.log_inference.remote(actor=self.name, gpu_ids=self.gpu_ids, stats=lora_stats)
                except Exception as exc:
                    self.logger.exception(f"failed to push inference stats -\n\n{exc}\n\n")

            await asyncio.sleep(3.0)

    def _tok_rate(self, lora: str, window: float = 2.0) -> float:
        """Rolling average tokens/sec over the last <window> seconds."""
        now  = time.monotonic()
        hist = self._tok_hist[lora]
        while hist and now - hist[0] > window:
            hist.popleft()
        return len(hist) / window
