import os, time, asyncio
from collections import defaultdict, deque
from typing import Optional, Dict, Any

import ray, torch
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

TOK_WINDOW = 100

@ray.remote
class VLLMActor:
    def __init__(self, vllm_config: Dict[str, Any], tracker, name):
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        torch.cuda.set_device(0)

        engine_args = EngineArgs(
            model=vllm_config["model_name"], enable_lora=True, max_loras=vllm_config.get("max_loras", 5),
            max_lora_rank=vllm_config["lora_config"]["lora_rank"], max_cpu_loras=vllm_config.get("max_loras", 5),
            max_num_seqs=vllm_config["max_parallel_seq"], task="generate", max_model_len=vllm_config.get("max_model_len", 8192)
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(
            temperature=vllm_config.get("temperature", 0.7), top_p=vllm_config.get("top_p", 0.95), max_tokens=vllm_config.get("max_tokens", 4096)
        )

        self._queue = deque()
        self._futures = {}
        self._next_id = 0
        self._req2lora = {}

        self.tracker = tracker 
        self.name = name 

        self._queued = defaultdict(int)
        self._running = defaultdict(int)
        self._tok_hist = defaultdict(lambda: deque(maxlen=TOK_WINDOW))

        loop = asyncio.get_event_loop()
        loop.create_task(self._batch_loop())
        loop.create_task(self._report_loop())

    async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
        fut = asyncio.Future()
        lora = lora_path or "base"
        self._queued[lora] += 1
        self._queue.append((prompt, lora_path, fut))
        return await fut

    async def _batch_loop(self):
        while True:
            await asyncio.sleep(0.02)
            while self._queue:
                prompt, path, fut = self._queue.popleft()
                lora = path or "base"
                req_id = str(self._next_id); self._next_id += 1
                self._futures[req_id] = fut
                self._req2lora[req_id] = lora

                self._queued[lora] -= 1
                self._running[lora] += 1

                lora_req = LoRARequest(path, path, abs(hash(path))) if path else None
                self.engine.add_request(req_id, prompt, self.sampling_params, lora_request=lora_req)

            for out in self.engine.step():
                lora = self._req2lora.get(out.request_id, "base")
                segment = out.outputs[-1]

                # Track tokens/sec
                if segment.token_ids:
                    now = time.monotonic()
                    for _ in segment.token_ids:
                        self._tok_hist[lora].append(now)

                # Resolve result only on finish
                if segment.finish_reason is not None:
                    fut = self._futures.pop(out.request_id, None)
                    if fut and not fut.done():
                        fut.set_result(segment.text)

                    self._running[lora] -= 1
                    self._req2lora.pop(out.request_id, None)

    async def _report_loop(self):
        while True:
            if self.tracker is not None:
                self.tracker.log_inference.remote(self.name, {lora: {"queued": self._queued[lora], "running": self._running[lora], "tok_s": self._tok_rate(lora)} for lora in set(list(self._queued) + list(self._running) + list(self._tok_hist))})
            await asyncio.sleep(3.0)

    def _tok_rate(self, lora: str, window: float = 2.0) -> float:
        now = time.monotonic()
        hist = self._tok_hist[lora]
        while hist and now - hist[0] > window:
            hist.popleft()
        return len(hist) / window
