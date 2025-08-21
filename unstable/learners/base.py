
import ray, torch, time, pathlib, os
import sys, traceback
from typing import List, Dict, Any, Optional
import GPUtil
from collections import deque

from unstable.buffers import BaseBuffer
from unstable.trackers import BaseTracker
from unstable.learners.utils import build_peft_model, enable_full_activation_ckpt
from unstable.utils import setup_logger


class BaseLearner:
    def __init__(self, model_name: str, lora_cfg: Dict[str,Any], batch_size: int, mini_batch_size: int, learning_rate: float, grad_clip: float, buffer: BaseBuffer, tracker: BaseTracker, model_registry, activation_checkpointing: bool=True, gradient_checkpointing: bool=True, use_trainer_cache: bool=False, initial_lora_path: Optional[str]=None): 
        # basically build the policy model and optimizer for policy model
        self.model_name, self.lora_cfg = model_name, lora_cfg
        self.buffer, self.tracker, self.model_registry = buffer, tracker, model_registry
        self.logger = setup_logger("learner", ray.get(tracker.get_log_dir.remote()))
        self.use_trainer_cache, self.gradient_checkpointing, self.activation_checkpointing = use_trainer_cache, gradient_checkpointing, activation_checkpointing
        self.batch_size, self.mini_batch_size, self.lr, self.grad_clip = batch_size, mini_batch_size, learning_rate, grad_clip
        self.gradient_acc_steps = self.batch_size // self.mini_batch_size # TODO maybe assert that divisible 
        self.ckpt_dir = pathlib.Path(ray.get(self.tracker.get_checkpoints_dir.remote())); self.ckpt_dir.mkdir(parents=True, exist_ok=True) # create ckpt dir

        torch.set_float32_matmul_precision('high')
        torch.set_default_dtype(torch.bfloat16)

        # 获取 Ray 分配的 GPU 并设置环境变量（与 VLLMActor 保持一致）
        ray_gpu_ids = ray.get_gpu_ids()
        if ray_gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray_gpu_ids))
            # 使用局部 GPU ID 0（因为 CUDA_VISIBLE_DEVICES 已经映射了物理 GPU）
            self.device = torch.device("cuda:0")
            # 对于监控，使用局部设备 ID 列表
            self.gpu_ids = list(range(len(ray_gpu_ids)))  # [0, 1, ...] 局部设备 ID
        else:
            self.device = torch.device("cpu")
            self.gpu_ids = []
        print(f"Using device: {self.device}, Ray GPU IDs: {ray_gpu_ids}, Local GPU IDs: {self.gpu_ids}")
        self.policy_model, self.tokenizer = build_peft_model(model_name, self.device, lora_cfg, initial_lora_path)
        self.policy_model.to(torch.bfloat16)

        if not self.use_trainer_cache:      self.policy_model.config.use_cache = False
        if self.gradient_checkpointing:     self.policy_model.gradient_checkpointing_enable() # gradient checkpointing
        if self.activation_checkpointing:   enable_full_activation_ckpt(self.policy_model)       # activation checkpointing. Affords most of the vRAM savings
        
        self.policy_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy_model.parameters()), lr=learning_rate)
        self._step = 1; self._samples_seen = 0 # training counters
        
        # GPU监控相关
        self._gpu_peak_memory = 0.0
        self._gpu_memory_samples = deque(maxlen=50)
        self._memory_pressure_threshold = 0.90  # 90%内存使用率阈值（训练时更严格）

    def _monitor_gpu_memory(self, phase: str = "unknown"):
        """监控GPU内存使用情况"""
        try:
            # 使用torch监控当前GPU内存
            if torch.cuda.is_available() and self.gpu_ids:
                for gpu_id in self.gpu_ids:
                    with torch.cuda.device(gpu_id):
                        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                        cached = torch.cuda.memory_reserved() / 1024**2      # MB
                        max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
                        
                        self._gpu_peak_memory = max(self._gpu_peak_memory, allocated)
                        self._gpu_memory_samples.append({
                            'timestamp': time.time(),
                            'phase': phase,
                            'step': self._step,
                            'gpu_id': gpu_id,
                            'allocated_mb': allocated,
                            'cached_mb': cached,
                            'max_allocated_mb': max_allocated
                        })
            
            # 使用GPUtil获取更详细的GPU信息
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.id in self.gpu_ids:
                    memory_util = gpu.memoryUtil
                    if memory_util > self._memory_pressure_threshold:
                        self.logger.warning(
                            f"High GPU memory usage during {phase} on GPU {gpu.id}: {memory_util*100:.1f}% "
                            f"({gpu.memoryUsed}MB/{gpu.memoryTotal}MB) at step {self._step}"
                        )
                        # 触发内存清理
                        self._cleanup_memory()
                        
        except Exception as e:
            self.logger.debug(f"GPU monitoring error during {phase}: {e}")
    
    def _cleanup_memory(self):
        """清理GPU内存"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 重置峰值内存统计
                for gpu_id in self.gpu_ids:
                    torch.cuda.reset_peak_memory_stats(gpu_id)
                self.logger.info(f"GPU cache cleared and peak memory stats reset at step {self._step}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup GPU memory: {e}")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """获取GPU统计信息"""
        stats = {
            'peak_memory_mb': self._gpu_peak_memory,
            'gpu_ids': self.gpu_ids,
            'current_step': self._step,
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

    def initialize_algorithm(self, cfg):    raise NotImplementedError
    def _update(self, batch):               raise NotImplementedError
    def train(self, iterations: int):
        self.logger.info("Starting training loop")
        
        # 初始GPU状态监控
        self._monitor_gpu_memory("training_start")

        while self._step < iterations:
            try:
                # 等待数据时监控GPU
                self._monitor_gpu_memory("waiting_for_data")
                while (ray.get(self.buffer.size.remote()) < self.batch_size * 1.5): 
                    time.sleep(0.2) # wait until enough data is available
                
                self.logger.info("Enough data, starting learning step")
                
                # 获取批次数据前监控
                self._monitor_gpu_memory("before_batch_load")
                batch: List = ray.get(self.buffer.get_batch.remote(self.batch_size)); self._samples_seen += self.batch_size
                
                # 更新前监控
                self._monitor_gpu_memory("before_update")
                accumulated_metrics = self._update(batch=batch) # handled by specific algo implementations
                
                # 更新后监控
                self._monitor_gpu_memory("after_update")

                # 添加GPU统计到日志
                gpu_stats = self.get_gpu_stats()
                log = {f"{k}": v for k, v in accumulated_metrics.items()}
                log.update({
                    "step": self._step,  
                    "samples_seen": self._samples_seen,  
                    "lr": self.policy_optimizer.param_groups[0]["lr"], 
                    "policy_grad_norm": sum(p.grad.data.norm(2).item()**2 for p in self.policy_model.parameters() if p.grad is not None) ** 0.5,
                    "gpu_peak_memory_mb": gpu_stats['peak_memory_mb'],
                    "gpu_current_allocated_mb": sum(gpu['allocated_mb'] for gpu in gpu_stats.get('current_memory', {}).values()),
                    "gpu_current_cached_mb": sum(gpu['cached_mb'] for gpu in gpu_stats.get('current_memory', {}).values())
                })
                self.tracker.log_learner.remote(log)

                # 保存检查点前监控
                self._monitor_gpu_memory("before_checkpoint_save")
                
                # save & register the updated checkpoint
                ckpt_path = self._save_checkpoint()
                try:
                    self.model_registry.add_checkpoint.remote(uid=f"ckpt-{self._step}", path=ckpt_path, iteration=self._step)
                    self.logger.info(f"Registered new ckpt: {ckpt_path}, ckpt-{self._step}")
                except Exception as exc: self.logger.info(f"Exception when adding checkpoint: {exc}")
                self.logger.info(f"registered new ckpt -> {ckpt_path} for iteration{self._step}")
                
                # 保存检查点后监控和清理
                self._monitor_gpu_memory("after_checkpoint_save")
                
                # 每10步进行一次内存清理
                if self._step % 10 == 0:
                    self._cleanup_memory()
                
                self._step += 1
            except Exception as exc:
                # 记录精确的报错位置（文件、行号、函数、代码行）
                exc_type, exc_value, exc_tb = sys.exc_info()
                last = traceback.extract_tb(exc_tb)[-1] if exc_tb else None
                if last is not None:
                    self.logger.error(
                        f"Exception in learner loop at {last.filename}:{last.lineno} in {last.name}: {exc}\n> {last.line}"
                    )
                else:
                    self.logger.error(f"Exception in learner loop: {exc}")
                # 输出完整堆栈
                self.logger.exception("Full traceback follows")

        # 训练结束时的GPU统计
        self._monitor_gpu_memory("training_end")
        final_stats = self.get_gpu_stats()
        self.logger.info(f"[Learner] training finished. Final GPU peak memory: {final_stats['peak_memory_mb']:.1f} MB")
        
        # 最终清理
        self._cleanup_memory()
        self.buffer.stop.remote()

    def _save_checkpoint(self):
        ckpt_dir = self.ckpt_dir / f"iteration-{self._step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.policy_model.save_pretrained(ckpt_dir, save_adapter=True)
        return ckpt_dir

