"""
测试脚本：监控VLLMActor和策略更新模块的GPU使用情况
"""

import sys
import os
import time
import asyncio
import ray
import torch
import logging
from pathlib import Path

# 添加项目路径
sys.path.append('/home/aiscuser/mindgames/UnstableBaselines')
sys.path.append('/home/aiscuser/mindgames/UnstableBaselines/test')

from gpu_monitor import GPUMonitor, GPUMemoryManager, gpu_monitor_context
from unstable.actor import VLLMActor
from unstable.learners.base import BaseLearner
from unstable.trackers import BaseTracker
from unstable.buffers import BaseBuffer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gpu_test")

class MockTracker:
    """模拟Tracker用于测试"""
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_log_dir(self):
        return str(self.log_dir)
    
    def get_checkpoints_dir(self):
        return str(self.log_dir / "checkpoints")
    
    def log_inference(self, actor: str, gpu_ids, stats):
        logger.info(f"[TRACKER] {actor} GPUs={gpu_ids} stats={stats}")
    
    def log_learner(self, log_data):
        logger.info(f"[TRACKER] Learner: {log_data}")

class MockBuffer:
    """模拟Buffer用于测试"""
    def __init__(self):
        self.data = []
        self._size = 0
    
    def size(self):
        return self._size
    
    def get_batch(self, batch_size):
        # 返回模拟批次数据
        return [{"obs": f"test_obs_{i}", "action": f"test_action_{i}"} for i in range(batch_size)]
    
    def stop(self):
        pass

async def test_vllm_actor_gpu_usage():
    """测试VLLMActor的GPU使用情况"""
    logger.info("=== Testing VLLMActor GPU Usage ===")
    
    # 初始化GPU监控器
    monitor = GPUMonitor(sample_interval=0.5, log_file="vllm_actor_gpu.log")
    memory_manager = GPUMemoryManager()
    
    try:
        monitor.start_monitoring()
        
        # 创建模拟tracker
        tracker = MockTracker("./test_logs/vllm_actor")
        
        # VLLMActor配置
        vllm_config = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",  # 使用较小的模型进行测试
            "max_loras": 2,
            "lora_config": {"lora_rank": 8},
            "max_parallel_seq": 2,
            "max_model_len": 1024,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 64,
            "logprobs_k": 5,
        }
        
        with gpu_monitor_context("vllm_actor_init", monitor):
            # 创建VLLMActor
            actor = VLLMActor.options(num_gpus=1).remote(
                cfg=vllm_config, 
                tracker=ray.put(tracker), 
                name="TestActor"
            )
        
        # 测试多个推理请求
        test_prompts = [
            "What is machine learning?",
            "Explain deep learning in simple terms.",
            "How does reinforcement learning work?",
            "What are the benefits of using GPUs for AI?",
            "Describe the transformer architecture."
        ]
        
        with gpu_monitor_context("vllm_inference", monitor):
            for i, prompt in enumerate(test_prompts):
                logger.info(f"Processing prompt {i+1}/{len(test_prompts)}")
                monitor.set_current_module(f"vllm_inference_prompt_{i+1}")
                
                # 执行推理
                result = await actor.submit_prompt.remote(prompt)
                text, cum_logp, gen_len = result
                
                logger.info(f"Prompt {i+1} result: {len(text)} chars, logp={cum_logp:.3f}, tokens={gen_len}")
                
                # 获取GPU统计
                gpu_stats = ray.get(actor.get_gpu_stats.remote())
                logger.info(f"GPU peak memory: {gpu_stats['peak_memory_mb']:.1f} MB")
                
                # 短暂等待以观察内存变化
                await asyncio.sleep(1)
        
        # 检查内存压力
        high_usage = memory_manager.check_memory_pressure(threshold_percent=70.0)
        if high_usage:
            logger.warning(f"High memory usage detected on GPUs: {high_usage}")
            memory_manager.optimize_memory_usage()
        
    finally:
        monitor.stop_monitoring()
        monitor.print_summary()
        monitor.save_detailed_report("vllm_actor_gpu_report.json")

def test_learner_gpu_usage():
    """测试Learner的GPU使用情况（模拟）"""
    logger.info("=== Testing Learner GPU Usage ===")
    
    # 初始化GPU监控器
    monitor = GPUMonitor(sample_interval=0.2, log_file="learner_gpu.log")
    memory_manager = GPUMemoryManager()
    
    try:
        monitor.start_monitoring()
        
        # 创建模拟组件
        tracker = MockTracker("./test_logs/learner")
        buffer = MockBuffer()
        buffer._size = 100  # 模拟有足够数据
        
        # 模拟Learner配置
        learner_config = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "lora_cfg": {
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            },
            "batch_size": 8,
            "mini_batch_size": 4,
            "learning_rate": 1e-4,
            "grad_clip": 1.0
        }
        
        with gpu_monitor_context("learner_init", monitor):
            # 这里我们模拟Learner的初始化和训练过程
            # 实际使用时会创建真实的Learner实例
            
            # 模拟模型加载
            logger.info("Simulating model loading...")
            if torch.cuda.is_available():
                # 创建一个小模型来模拟内存使用
                model = torch.nn.Sequential(
                    torch.nn.Linear(512, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 256)
                ).cuda()
                
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 模拟训练步骤
        for step in range(5):
            with gpu_monitor_context(f"training_step_{step+1}", monitor):
                logger.info(f"Simulating training step {step+1}")
                
                if torch.cuda.is_available():
                    # 模拟前向传播
                    monitor.set_current_module(f"forward_pass_step_{step+1}")
                    batch_data = torch.randn(8, 512).cuda()
                    output = model(batch_data)
                    
                    # 模拟损失计算和反向传播
                    monitor.set_current_module(f"backward_pass_step_{step+1}")
                    target = torch.randn_like(output)
                    loss = torch.nn.functional.mse_loss(output, target)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    logger.info(f"Step {step+1} loss: {loss.item():.4f}")
                
                # 检查内存压力
                high_usage = memory_manager.check_memory_pressure(threshold_percent=80.0)
                if high_usage:
                    logger.warning(f"High memory usage at step {step+1}")
                    memory_manager.optimize_memory_usage()
                
                time.sleep(2)  # 模拟训练时间
        
    finally:
        monitor.stop_monitoring()
        monitor.print_summary()
        monitor.save_detailed_report("learner_gpu_report.json")

def test_memory_pressure_relief():
    """测试内存压力缓解策略"""
    logger.info("=== Testing Memory Pressure Relief Strategies ===")
    
    memory_manager = GPUMemoryManager()
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping memory pressure test")
        return
    
    # 获取初始内存状态
    initial_memory = memory_manager.get_memory_info()
    logger.info(f"Initial GPU memory: {initial_memory}")
    
    try:
        # 创建大量张量以增加内存压力
        tensors = []
        logger.info("Creating tensors to increase memory pressure...")
        
        for i in range(10):
            # 创建较大的张量
            tensor = torch.randn(1000, 1000).cuda()
            tensors.append(tensor)
            
            # 检查内存使用
            current_memory = memory_manager.get_memory_info()
            for gpu_id, info in current_memory.items():
                logger.info(f"GPU {gpu_id}: Allocated {info['allocated']:.1f} MB, Cached {info['cached']:.1f} MB")
            
            # 检查是否需要清理内存
            high_usage = memory_manager.check_memory_pressure(threshold_percent=70.0)
            if high_usage:
                logger.info("Triggering memory optimization...")
                memory_manager.optimize_memory_usage()
                
                # 删除一些张量
                if len(tensors) > 5:
                    del tensors[:3]
                    torch.cuda.empty_cache()
                    logger.info("Freed some tensors")
        
        # 最终内存状态
        final_memory = memory_manager.get_memory_info()
        logger.info(f"Final GPU memory: {final_memory}")
        
    except Exception as e:
        logger.error(f"Error during memory pressure test: {e}")
    
    finally:
        # 清理所有张量
        if 'tensors' in locals():
            del tensors
        memory_manager.optimize_memory_usage()
        logger.info("Memory pressure test completed and cleaned up")

async def main():
    """主测试函数"""
    logger.info("Starting GPU monitoring tests...")
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    try:
        # 测试1: VLLMActor GPU使用情况
        await test_vllm_actor_gpu_usage()
        
        # 等待一段时间让GPU内存稳定
        logger.info("Waiting for GPU memory to stabilize...")
        time.sleep(5)
        
        # 测试2: Learner GPU使用情况
        test_learner_gpu_usage()
        
        # 等待一段时间
        logger.info("Waiting for GPU memory to stabilize...")
        time.sleep(5)
        
        # 测试3: 内存压力缓解
        test_memory_pressure_relief()
        
        logger.info("All GPU monitoring tests completed!")
        
        # 生成综合报告
        logger.info("\n" + "="*80)
        logger.info("GPU MONITORING TEST SUMMARY")
        logger.info("="*80)
        logger.info("1. VLLMActor GPU monitoring: ✓ Completed")
        logger.info("2. Learner GPU monitoring: ✓ Completed") 
        logger.info("3. Memory pressure relief: ✓ Completed")
        logger.info("Reports saved:")
        logger.info("  - vllm_actor_gpu_report.json")
        logger.info("  - learner_gpu_report.json")
        logger.info("  - vllm_actor_gpu.log")
        logger.info("  - learner_gpu.log")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    
    finally:
        # 清理Ray资源
        ray.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
