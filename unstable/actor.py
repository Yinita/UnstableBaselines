
import os, ray, asyncio, time, collections
from typing import Any, Dict, Optional
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest
from unstable.utils.logging import setup_logger


@ray.remote
class VLLMActor:
    def __init__(self, vllm_config: Dict[str, Any], tracker, name: str):
        self.logger = setup_logger(f"actor-{name}", ray.get(tracker.get_log_dir.remote()))
        self.gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        self.engine = LLMEngine.from_engine_args(EngineArgs(
            model=vllm_config["model_name"], enable_lora=True, max_loras=vllm_config["max_loras"], max_lora_rank=vllm_config["lora_config"]["lora_rank"],
            max_num_seqs=vllm_config["max_parallel_seq"], max_model_len=vllm_config["max_model_len"], task="generate", disable_log_stats=True,  # silence vllm's own metrics
        ))
        self.sampling = SamplingParams(temperature=vllm_config.get("temperature", 0.7), top_p=vllm_config.get("top_p", 0.95), max_tokens=vllm_config.get("max_tokens", 512),)

        self.queued: int = 0
        self.running: int = 0
        self._tok_hist: collections.deque[float] = collections.deque()
        self._futs: Dict[str, asyncio.Future] = {}
        self._next_id = 0
        self.tracker = tracker
        self.name = name

        loop = asyncio.get_event_loop()
        loop.create_task(self._batch_loop())
        loop.create_task(self._report_loop())

    async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> str:
        fut = asyncio.Future()
        req_id = str(self._next_id); self._next_id += 1
        lora_req = None
        if lora_path:
            lora_req = LoRARequest(lora_path, 1, lora_path)  # id=1 always fine here

        self.engine.add_request(req_id, prompt, self.sampling, lora_request=lora_req)
        self._futs[req_id] = fut
        self.queued += 1
        return await fut

    # ───────────────────────── internal tasks ──────────────────────────────
    async def _batch_loop(self):
        while True:
            outs = self.engine.step()
            if self.queued:
                self.running += self.queued
                self.queued = 0

            now = time.monotonic()
            for out in outs:
                seg = out.outputs[-1]
                new_tok = len(seg.token_ids)
                self._tok_hist.extend([now] * new_tok)

                if seg.finish_reason is not None:
                    fut = self._futs.pop(out.request_id, None)
                    if fut and not fut.done():
                        fut.set_result(seg.text)
                    self.running -= 1

    async def _report_loop(self):
        while True:
            await asyncio.sleep(5.0) # only send every 5 sec
            tok_s = self._tok_rate()
            stats = {"queued": self.queued, "running": self.running, "tok_s": tok_s}
            try:
                await self.tracker.log_inference.remote(actor=self.name, gpu_ids=self.gpu_ids, stats=stats)
            except Exception as e:
                self.logger.warning(f"tracker logging failed: {e}")

    def _tok_rate(self, window: float = 2.0) -> float:
        """Rolling tokens/sec across all requests."""
        now = time.monotonic()
        while self._tok_hist and now - self._tok_hist[0] > window:
            self._tok_hist.popleft()
        return len(self._tok_hist) / window

