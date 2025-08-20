import os, time, asyncio
from collections import defaultdict, deque
from typing import Optional, Dict, Any, Tuple  # 新增 Tuple

import ray
from vllm import LLM, EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest
import torch
import GPUtil

from unstable.utils.logging import setup_logger


@ray.remote(max_concurrency=1000)
class VLLMActor:
    def __init__(self, cfg: Dict[str, Any], tracker, name: str):
        # Handle tracker reference - could be Ray actor, object reference, or direct object
        try:
            # Check if tracker is an ObjectRef from ray.put()
            if hasattr(tracker, '_id'):  # ObjectRef has _id attribute
                tracker_obj = ray.get(tracker)
                log_dir = tracker_obj.get_log_dir()
            elif hasattr(tracker, 'remote'):
                # tracker is a Ray actor
                log_dir = ray.get(tracker.get_log_dir.remote())
            elif hasattr(tracker, 'get_log_dir'):
                # tracker is a direct object (like MockTracker)
                log_dir = tracker.get_log_dir()
            else:
                raise ValueError(f"Unknown tracker type: {type(tracker)}")
        except Exception as e:
            # Fallback to a default log directory if tracker handling fails
            log_dir = "./logs"
            os.makedirs(log_dir, exist_ok=True)
        
        self.logger = setup_logger(f"actor-{name}", log_dir) # set up logging
        self.gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        
        # 使用高层 LLM 接口，便于一次性批量 generate
        try:
            tp = max(1, len(self.gpu_ids) if isinstance(self.gpu_ids, (list, tuple)) else 1)
            self.inference_engine = LLM(
                model=cfg["model_name"],
                enable_lora=True,
                max_loras=cfg["max_loras"],
                max_lora_rank=cfg["lora_config"]["lora_rank"],
                max_num_seqs=cfg["max_parallel_seq"],
                max_model_len=cfg["max_model_len"],
                gpu_memory_utilization=cfg["gpu_memory_utilization"],
                tensor_parallel_size=tp,
                trust_remote_code=True,
            )
            self.logger.info("vLLM LLM initialized successfully")
        except Exception as e:
            self.logger.error(f"vLLM LLM initialization failed: {e}");
            raise
        self.logger.info(f"vLLM model path or name: {cfg['model_name']}")
        # 批大小上限（用于 _batch_loop）
        try:
            self._max_bs = int(max(1, cfg.get("max_parallel_seq", 1)))
        except Exception:
            self._max_bs = 1
            
        self.sampling_params = SamplingParams(
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.95),
            max_tokens=cfg.get("max_tokens", 4096),
            # logprobs=cfg.get("logprobs_k", 1),    # 新增：回传每步 top-k logprobs
            # prompt_logprobs=0                     # 仅需要生成部分的 logprobs
        )

        self._queue = deque()
        self._futures = {}
        self._next_id = 0
        self._req2lora = {}
        self._prev_tok_cnt = defaultdict(int)

        self.tracker = tracker
        self.name = name

        self._queued = 0
        self._running = 0
        self._tok_hist = deque()
        self._batch_task = asyncio.create_task(self._batch_loop())
        self._report_task = asyncio.create_task(self._report_loop())
        self._lora_ids: Dict[str, int] = {"base": 0}
        self._next_lora_id = 1
        self._last_step_time = time.monotonic()  # Add health check flag
        
        # GPU监控相关
        self._gpu_peak_memory = 0.0
        self._gpu_memory_samples = deque(maxlen=100)
        self._memory_pressure_threshold = 0.85  # 85%内存使用率阈值

    def _resolve_lora_path(self, path: str) -> Optional[str]:
        """
        尝试多种方式解析LoRA路径，处理相对路径和绝对路径的问题
        """
        if not path:
            return None
            
        # 策略1: 直接使用原路径（如果是绝对路径）
        if os.path.isabs(path) and os.path.exists(path):
            return path
            
        # 策略2: 在当前工作目录下查找
        cwd_path = os.path.join(os.getcwd(), path)
        if os.path.exists(cwd_path):
            return cwd_path
            
        # 策略3: 在outputs目录下查找最新的匹配目录
        outputs_base = "/home/aiscuser/mindgames/outputs"
        if os.path.exists(outputs_base):
            # 获取所有日期目录，按时间排序
            date_dirs = []
            for date_dir in os.listdir(outputs_base):
                date_path = os.path.join(outputs_base, date_dir)
                if os.path.isdir(date_path):
                    date_dirs.append((date_dir, date_path))
            
            # 按日期降序排序
            date_dirs.sort(reverse=True)
            
            for date_dir, date_path in date_dirs:
                # 在每个日期目录下查找时间目录
                time_dirs = []
                for time_dir in os.listdir(date_path):
                    time_path = os.path.join(date_path, time_dir)
                    if os.path.isdir(time_path):
                        time_dirs.append((time_dir, time_path))
                
                # 按时间降序排序
                time_dirs.sort(reverse=True)
                
                for time_dir, time_path in time_dirs:
                    # 在每个时间目录下查找匹配的路径
                    potential_path = os.path.join(time_path, path)
                    if os.path.exists(potential_path):
                        return potential_path
                    
                    # 也尝试在子目录中查找
                    for subdir in os.listdir(time_path):
                        subdir_path = os.path.join(time_path, subdir)
                        if os.path.isdir(subdir_path):
                            potential_path = os.path.join(subdir_path, path)
                            if os.path.exists(potential_path):
                                return potential_path
        
        # 策略4: 如果路径包含特定的标识符，尝试模式匹配
        if "base_" in path or "ckpt-" in path:
            # 尝试在所有可能的checkpoint目录中查找
            for root, dirs, files in os.walk("/home/aiscuser/mindgames/outputs"):
                if "checkpoints" in root or "iteration-" in root:
                    potential_path = os.path.join(root, os.path.basename(path))
                    if os.path.exists(potential_path):
                        return potential_path
        
        return None

    async def submit_prompt(self, prompt: str, lora_path: Optional[str] = None) -> Tuple[str, float, int]:
        if lora_path is not None and not isinstance(lora_path, str):
            lora_path = str(lora_path)
        fut = asyncio.Future()
        self._queued += 1
        self._queue.append((prompt, lora_path, fut))
        # 在 batch_loop 完成时，fut 会被设置为 (text, cum_logp, gen_len)
        return await fut

    def _monitor_gpu_memory(self):
        """轻量级监控：仅用 torch 采样显存，避免热路径中调用 GPUtil。"""
        try:
            if torch.cuda.is_available() and self.gpu_ids:
                for gpu_id in self.gpu_ids:
                    with torch.cuda.device(gpu_id):
                        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                        cached = torch.cuda.memory_reserved() / 1024**2      # MB

                        self._gpu_peak_memory = max(self._gpu_peak_memory, allocated)
                        self._gpu_memory_samples.append({
                            'timestamp': time.time(),
                            'gpu_id': gpu_id,
                            'allocated_mb': allocated,
                            'cached_mb': cached
                        })
        except Exception as e:
            self.logger.debug(f"GPU monitoring (light) error: {e}")

    def _monitor_gpu_memory_heavy(self):
        """重监控：使用 GPUtil 获取详细信息，仅在 report loop 中低频调用。"""
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if self.gpu_ids and gpu.id in self.gpu_ids:
                    memory_util = gpu.memoryUtil
                    if memory_util > self._memory_pressure_threshold:
                        self.logger.warning(
                            f"High GPU memory usage on GPU {gpu.id}: {memory_util*100:.1f}% "
                            f"({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)"
                        )
                        self._cleanup_memory()
        except Exception as e:
            self.logger.debug(f"GPU monitoring (heavy) error: {e}")
    
    def _cleanup_memory(self):
        """清理GPU内存"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("GPU cache cleared due to memory pressure")
        except Exception as e:
            self.logger.error(f"Failed to cleanup GPU memory: {e}")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """获取GPU统计信息"""
        stats = {
            'peak_memory_mb': self._gpu_peak_memory,
            'gpu_ids': self.gpu_ids,
            'recent_samples': list(self._gpu_memory_samples)[-10:] if self._gpu_memory_samples else []
        }
        
        # 添加当前内存使用情况
        if torch.cuda.is_available() and self.gpu_ids:
            current_memory = {}
            for gpu_id in self.gpu_ids:
                with torch.cuda.device(gpu_id):
                    current_memory[f'gpu_{gpu_id}'] = {
                        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                        'cached_mb': torch.cuda.memory_reserved() / 1024**2,
                        'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2
                    }
            stats['current_memory'] = current_memory
        
        return stats

    async def _batch_loop(self):
        while True:
            try:
                await asyncio.sleep(0.02)

                if time.monotonic() - self._last_step_time > 30:
                    self.logger.error(
                        f"Potential deadlock detected - no engine steps for {time.monotonic() - self._last_step_time:.1f} seconds\nRunning requests: {self._running}\nQueue size: {len(self._queue)}"
                    )

                # 尝试组一个 batch
                if not self._queue:
                    continue

                batch_prompts = []
                batch_loras = []
                batch_req_ids = []
                batch_futs = []
                max_bs = self._max_bs

                while self._queue and len(batch_prompts) < max_bs:
                    prompt, path, fut = self._queue.popleft()
                    req_id = str(self._next_id); self._next_id += 1
                    self._futures[req_id] = fut
                    self._queued -= 1
                    self._running += 1
                    batch_futs.append(fut)
                    batch_req_ids.append(req_id)

                    # LoRA 解析
                    if path:
                        resolved_path = self._resolve_lora_path(path)
                        if resolved_path and os.path.exists(resolved_path):
                            if resolved_path not in self._lora_ids:
                                self._lora_ids[resolved_path] = self._next_lora_id
                                self._next_lora_id += 1
                            lora_req = LoRARequest(resolved_path, self._lora_ids[resolved_path], resolved_path)
                        else:
                            self.logger.warning(
                                f"LoRA path does not exist or is invalid: {path} (resolved: {resolved_path}), using base model instead"
                            )
                            lora_req = None
                    else:
                        lora_req = None

                    batch_prompts.append(prompt)
                    batch_loras.append(lora_req)

                # 生成（放到线程里，避免阻塞事件循环）
                step_start = time.monotonic()
                try:
                    outs = await asyncio.to_thread(
                        self.inference_engine.generate,
                        batch_prompts,
                        self.sampling_params,
                        lora_request=batch_loras,
                        use_tqdm=False,
                    )
                except Exception as e:
                    self.logger.exception(f"LLM.generate failed for batch size={len(batch_prompts)}: {e}")
                    # fail fast for this batch
                    for req_id, fut in zip(batch_req_ids, batch_futs):
                        if not fut.done():
                            fut.set_exception(e)
                        self._running -= 1
                        self._prev_tok_cnt.pop(req_id, None)
                    continue

                step_end = time.monotonic()
                step_duration = step_end - step_start
                self._last_step_time = step_end
                if step_duration > 5.0:
                    self.logger.warning(f"Slow generate batch: {step_duration:.1f}s size={len(batch_prompts)}")

                # 处理输出，逐个完成 future
                total_new_tokens = 0
                for req_id, fut, out in zip(batch_req_ids, batch_futs, outs):
                    try:
                        segment = out.outputs[-1]
                        tok_ids = getattr(segment, "token_ids", None) or []
                        gen_len = len(tok_ids)
                        total_new_tokens += gen_len

                        cum_logp = getattr(segment, "cumulative_logprob", None)
                        if cum_logp is None:
                            cum = 0.0
                            seg_logprobs = getattr(segment, "logprobs", None) or []
                            for step_lp in seg_logprobs:
                                if not step_lp:
                                    continue
                                chosen_lp = None
                                for _, lp in step_lp.items():
                                    if getattr(lp, "rank", None) == 0:
                                        chosen_lp = lp
                                        break
                                if chosen_lp is None:
                                    vals = []
                                    for _, lp in step_lp.items():
                                        try:
                                            vals.append(float(getattr(lp, "logprob", lp)))
                                        except (ValueError, TypeError):
                                            continue
                                    if vals:
                                        cum += max(vals)
                                else:
                                    try:
                                        cum += float(getattr(chosen_lp, "logprob", chosen_lp))
                                    except (ValueError, TypeError):
                                        pass
                            cum_logp = float(cum)
                        else:
                            cum_logp = float(cum_logp)

                        if not fut.done():
                            fut.set_result((segment.text, cum_logp, gen_len))
                    except Exception as e:
                        if not fut.done():
                            fut.set_exception(e)
                    finally:
                        self._running -= 1
                        self._prev_tok_cnt.pop(req_id, None)
                        self._req2lora.pop(req_id, None)

                # 更新 tok_s 统计：将本批生成的 token 均匀分布到 step_duration 内
                if total_new_tokens > 0:
                    if step_duration <= 0:
                        # 退化：全部记在当前时刻
                        for _ in range(total_new_tokens):
                            self._tok_hist.append(step_end)
                    else:
                        dt = step_duration / total_new_tokens
                        t = step_end - step_duration
                        for _ in range(total_new_tokens):
                            self._tok_hist.append(t)
                            t += dt
            except Exception as e:
                self.logger.exception(f"Critical error in batch loop: {e}")
                await asyncio.sleep(1.0)  # Prevent tight error loop
    
    async def _report_loop(self):
        self.logger.info("Starting _report_loop")
        itr = 0
        while True:
            loop_start = time.monotonic()
            await asyncio.sleep(5.0)  # only send every 5 sec
            itr += 1

            now = time.monotonic()
            iter_elapsed = now - loop_start
            last_step_age = now - self._last_step_time

            # 低频 GPU 监控：轻量每 5s；重监控每 ~10s
            self._monitor_gpu_memory()
            if itr % 2 == 0:
                self._monitor_gpu_memory_heavy()

            # Lightweight snapshots
            try:
                ray_res = ray.available_resources()
            except Exception:
                ray_res = {}

            try:
                gpu_stats = self.get_gpu_stats()
            except Exception as e:
                gpu_stats = {"error": str(e)}

            stats = {
                "queued": self._queued,
                "queue_len": len(self._queue),
                "running": self._running,
                "futures": len(self._futures),
                "tok_s": self._tok_rate(),
                "iter": itr,
                "iter_elapsed_s": round(iter_elapsed, 3),
                "last_step_age_s": round(last_step_age, 3),
                "ray": {
                    # Only keep a small, stable subset
                    "CPU": float(ray_res.get("CPU", 0.0)) if isinstance(ray_res.get("CPU", 0.0), (int, float)) else 0.0,
                    "GPU": float(ray_res.get("GPU", 0.0)) if isinstance(ray_res.get("GPU", 0.0), (int, float)) else 0.0,
                },
                "gpu": {
                    "ids": self.gpu_ids,
                    "peak_mb": gpu_stats.get("peak_memory_mb", None) if isinstance(gpu_stats, dict) else None,
                    "recent_samples": gpu_stats.get("recent_samples", [])[-3:] if isinstance(gpu_stats, dict) else [],
                },
            }

            # Log a concise line for quick grepping
            self.logger.info(
                "report: iter=%d queued=%d queue_len=%d running=%d futures=%d tok_s=%.2f last_step_age=%.2fs gpu_ids=%s peak=%.1fMB",
                stats["iter"], stats["queued"], stats["queue_len"], stats["running"], stats["futures"],
                stats["tok_s"], stats["last_step_age_s"], str(stats["gpu"]["ids"]),
                (stats["gpu"]["peak_mb"] or 0.0),
            )

            # Emit a warning if the engine hasn't stepped in a while
            if last_step_age > 30.0:
                self.logger.warning(
                    f"Engine has not stepped for {last_step_age:.1f}s. queued={self._queued} queue_len={len(self._queue)} running={self._running}"
                )

            # Best-effort tracker push
            try:
                await self.tracker.log_inference.remote(actor=self.name, gpu_ids=self.gpu_ids, stats=stats)
            except Exception as e:
                self.logger.warning(f"tracker logging failed: {e} | tracker_type={type(self.tracker)}")

    def _tok_rate(self, window: float = 2.0) -> float:
        now  = time.monotonic()
        while self._tok_hist and now - self._tok_hist[0] > window:
            self._tok_hist.popleft()
        return len(self._tok_hist) / window


if __name__ == "__main__":
    import asyncio
    import ray

    # 一个极简 tracker actor：满足 VLLMActor 里用到的两个接口
    @ray.remote
    class DummyTracker:
        def __init__(self):
            self._logdir = os.path.abspath("./vllm_actor_logs")
            os.makedirs(self._logdir, exist_ok=True)
        def get_log_dir(self):
            return self._logdir
        def log_inference(self, actor: str, gpu_ids, stats: Dict[str, Any]):
            # 简单打印一下
            print(f"[TRACKER] {actor} GPUs={gpu_ids} stats={stats}")

    async def main():
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_gpus=4)
        print(ray.available_resources())
        tracker = DummyTracker.remote()

        # cfg = {
        #     "model_name": "Qwen/Qwen3-1.7B-Base",  # 你指定的测试模型
        #     "max_loras": 2,
        #     "lora_config": {"lora_rank": 8},
        #     "max_parallel_seq": 1,
        #     "max_model_len": 4096,
        #     "temperature": 0.7,
        #     "top_p": 0.95,
        #     "max_tokens": 4096,
        #     "logprobs_k": 5,  # 返回 top-5 logprobs
        #     "gpu_memory_utilization": 0.8
        # }
        cfg = {
        "model_name": "Qwen/Qwen3-1.7B-Base",
        "temperature": 0.6,
        "max_tokens":   4096,
        "max_parallel_seq": 1024,
        "max_loras": 2,
        "lora_config": {"lora_rank": 8},
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.8
    }

        # 启动 actor
        actor = VLLMActor.options(num_gpus=4).remote(cfg, tracker, "Actor-0")

        # 构造 20 条请求：10 条一样 + 10 条不一样
        base = "Write a long story about sunrise over mountains."
        same_prompts = [base for _ in range(10)]
        diff_prompts = [f"Write a long story about sunrise over mountains. Variant #{i}." for i in range(10)]
        prompts = same_prompts + diff_prompts

        # 并发提交并等待 (text, cum_logp, gen_len)
        refs = [actor.submit_prompt.remote(p) for p in prompts]
        results = await asyncio.gather(*refs)

        print("\n=== BATCH RESULTS (20) ===")
        for i, (text, cum_logp, gen_len) in enumerate(results):
            print(f"--- Result #{i} ---")
            print("CUM_LOGP:", cum_logp, "GEN_LEN:", gen_len)
            # 只打印前 200 字符，避免刷屏
            preview = text[:100].replace("\n", " ")
            print("TEXT:", preview + ("..." if len(text) > 200 else ""))

        # 关闭 ray
        ray.shutdown()

    # 运行
    asyncio.run(main())
