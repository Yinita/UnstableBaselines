"""
内存优化策略模块 - 提供多种GPU内存压力缓解方案
"""

import torch
import gc
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import GPUtil

logger = logging.getLogger(__name__)

class MemoryPressureLevel(Enum):
    """内存压力等级"""
    LOW = "low"           # < 70%
    MEDIUM = "medium"     # 70-85%
    HIGH = "high"         # 85-95%
    CRITICAL = "critical" # > 95%

@dataclass
class MemoryOptimizationConfig:
    """内存优化配置"""
    # 压力阈值设置
    medium_threshold: float = 0.70
    high_threshold: float = 0.85
    critical_threshold: float = 0.95
    
    # 优化策略开关
    enable_cache_clearing: bool = True
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    enable_model_sharding: bool = False
    enable_cpu_offloading: bool = False
    
    # 批次大小调整
    enable_dynamic_batch_size: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 32
    batch_size_reduction_factor: float = 0.75
    
    # LoRA管理
    max_concurrent_loras: int = 3
    lora_cache_size: int = 5

class MemoryOptimizer:
    """GPU内存优化器"""
    
    def __init__(self, config: MemoryOptimizationConfig = None):
        self.config = config or MemoryOptimizationConfig()
        self.logger = logging.getLogger("memory_optimizer")
        
        # 状态跟踪
        self.current_batch_size = self.config.max_batch_size
        self.optimization_history = []
        self.active_loras = {}
        self.lora_cache = {}
        
    def get_memory_pressure_level(self, gpu_id: int = 0) -> MemoryPressureLevel:
        """获取当前内存压力等级"""
        try:
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                memory_util = gpus[gpu_id].memoryUtil
                
                if memory_util >= self.config.critical_threshold:
                    return MemoryPressureLevel.CRITICAL
                elif memory_util >= self.config.high_threshold:
                    return MemoryPressureLevel.HIGH
                elif memory_util >= self.config.medium_threshold:
                    return MemoryPressureLevel.MEDIUM
                else:
                    return MemoryPressureLevel.LOW
        except Exception as e:
            self.logger.error(f"Failed to get memory pressure level: {e}")
            return MemoryPressureLevel.LOW
    
    def clear_gpu_cache(self):
        """清理GPU缓存"""
        if not self.config.enable_cache_clearing:
            return
            
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()  # 也清理Python垃圾回收
                self.logger.info("GPU cache and Python garbage collection completed")
        except Exception as e:
            self.logger.error(f"Failed to clear GPU cache: {e}")
    
    def optimize_batch_size(self, current_batch_size: int, pressure_level: MemoryPressureLevel) -> int:
        """根据内存压力动态调整批次大小"""
        if not self.config.enable_dynamic_batch_size:
            return current_batch_size
            
        if pressure_level == MemoryPressureLevel.CRITICAL:
            new_batch_size = max(self.config.min_batch_size, int(current_batch_size * 0.5))
        elif pressure_level == MemoryPressureLevel.HIGH:
            new_batch_size = max(self.config.min_batch_size, int(current_batch_size * self.config.batch_size_reduction_factor))
        elif pressure_level == MemoryPressureLevel.LOW and current_batch_size < self.config.max_batch_size:
            new_batch_size = min(self.config.max_batch_size, int(current_batch_size * 1.25))
        else:
            new_batch_size = current_batch_size
            
        if new_batch_size != current_batch_size:
            self.logger.info(f"Adjusted batch size: {current_batch_size} -> {new_batch_size} (pressure: {pressure_level.value})")
            
        return new_batch_size
    
    def manage_lora_cache(self, requested_lora: str, active_loras: Dict[str, Any]):
        """管理LoRA缓存，避免同时加载过多LoRA"""
        if len(active_loras) >= self.config.max_concurrent_loras:
            # 移除最久未使用的LoRA
            oldest_lora = min(active_loras.keys(), key=lambda x: active_loras[x].get('last_used', 0))
            
            # 将其移到缓存中
            if len(self.lora_cache) >= self.config.lora_cache_size:
                # 缓存已满，删除最老的
                oldest_cached = min(self.lora_cache.keys(), key=lambda x: self.lora_cache[x].get('cached_time', 0))
                del self.lora_cache[oldest_cached]
                self.logger.info(f"Removed LoRA from cache: {oldest_cached}")
            
            self.lora_cache[oldest_lora] = {
                'data': active_loras[oldest_lora],
                'cached_time': time.time()
            }
            del active_loras[oldest_lora]
            self.logger.info(f"Moved LoRA to cache: {oldest_lora}")
        
        # 检查请求的LoRA是否在缓存中
        if requested_lora in self.lora_cache:
            active_loras[requested_lora] = self.lora_cache[requested_lora]['data']
            del self.lora_cache[requested_lora]
            self.logger.info(f"Restored LoRA from cache: {requested_lora}")
    
    def apply_gradient_checkpointing(self, model):
        """应用梯度检查点以节省内存"""
        if not self.config.enable_gradient_checkpointing:
            return
            
        try:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing enabled")
        except Exception as e:
            self.logger.error(f"Failed to enable gradient checkpointing: {e}")
    
    def optimize_for_pressure_level(self, pressure_level: MemoryPressureLevel, context: Dict[str, Any] = None):
        """根据压力等级应用相应的优化策略"""
        context = context or {}
        optimizations_applied = []
        
        if pressure_level == MemoryPressureLevel.CRITICAL:
            # 关键级别：应用所有可能的优化
            self.clear_gpu_cache()
            optimizations_applied.append("cache_clearing")
            
            # 大幅减少批次大小
            if 'current_batch_size' in context:
                new_batch_size = self.optimize_batch_size(context['current_batch_size'], pressure_level)
                context['suggested_batch_size'] = new_batch_size
                optimizations_applied.append("batch_size_reduction")
            
            # 清理LoRA缓存
            if hasattr(self, 'lora_cache'):
                self.lora_cache.clear()
                optimizations_applied.append("lora_cache_clearing")
                
        elif pressure_level == MemoryPressureLevel.HIGH:
            # 高级别：应用中等程度优化
            self.clear_gpu_cache()
            optimizations_applied.append("cache_clearing")
            
            if 'current_batch_size' in context:
                new_batch_size = self.optimize_batch_size(context['current_batch_size'], pressure_level)
                context['suggested_batch_size'] = new_batch_size
                optimizations_applied.append("batch_size_adjustment")
                
        elif pressure_level == MemoryPressureLevel.MEDIUM:
            # 中等级别：轻度优化
            self.clear_gpu_cache()
            optimizations_applied.append("cache_clearing")
            
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': time.time(),
            'pressure_level': pressure_level.value,
            'optimizations': optimizations_applied,
            'context': context.copy()
        })
        
        self.logger.info(f"Applied optimizations for {pressure_level.value} pressure: {optimizations_applied}")
        return optimizations_applied
    
    def get_memory_recommendations(self, gpu_id: int = 0) -> Dict[str, Any]:
        """获取内存优化建议"""
        pressure_level = self.get_memory_pressure_level(gpu_id)
        
        recommendations = {
            'pressure_level': pressure_level.value,
            'immediate_actions': [],
            'configuration_suggestions': [],
            'monitoring_alerts': []
        }
        
        if pressure_level == MemoryPressureLevel.CRITICAL:
            recommendations['immediate_actions'].extend([
                "Reduce batch size immediately",
                "Clear GPU cache",
                "Consider model sharding",
                "Enable CPU offloading if available"
            ])
            recommendations['configuration_suggestions'].extend([
                "Enable gradient checkpointing",
                "Use mixed precision training",
                "Reduce max sequence length",
                "Limit concurrent LoRA adapters"
            ])
            recommendations['monitoring_alerts'].append("CRITICAL: GPU memory usage > 95%")
            
        elif pressure_level == MemoryPressureLevel.HIGH:
            recommendations['immediate_actions'].extend([
                "Clear GPU cache",
                "Reduce batch size",
                "Limit LoRA cache size"
            ])
            recommendations['configuration_suggestions'].extend([
                "Enable gradient checkpointing",
                "Consider mixed precision"
            ])
            recommendations['monitoring_alerts'].append("WARNING: High GPU memory usage")
            
        elif pressure_level == MemoryPressureLevel.MEDIUM:
            recommendations['immediate_actions'].append("Monitor memory usage closely")
            recommendations['configuration_suggestions'].append("Prepare for potential optimizations")
            
        return recommendations
    
    def create_memory_efficient_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建内存高效的配置"""
        efficient_config = base_config.copy()
        
        # 调整批次大小
        if 'batch_size' in efficient_config:
            efficient_config['batch_size'] = min(efficient_config['batch_size'], 8)
        
        # 调整序列长度
        if 'max_model_len' in efficient_config:
            efficient_config['max_model_len'] = min(efficient_config['max_model_len'], 2048)
        
        # 调整并行序列数
        if 'max_parallel_seq' in efficient_config:
            efficient_config['max_parallel_seq'] = min(efficient_config['max_parallel_seq'], 4)
        
        # 启用内存优化选项
        efficient_config.update({
            'enable_gradient_checkpointing': True,
            'enable_activation_checkpointing': True,
            'use_trainer_cache': False,
            'max_loras': min(efficient_config.get('max_loras', 5), 3)
        })
        
        return efficient_config

# 装饰器：自动内存优化
def auto_memory_optimize(optimizer: MemoryOptimizer = None):
    """装饰器：自动进行内存优化"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            if optimizer is None:
                local_optimizer = MemoryOptimizer()
            else:
                local_optimizer = optimizer
            
            # 检查内存压力
            pressure_level = local_optimizer.get_memory_pressure_level()
            
            # 如果压力过高，先进行优化
            if pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                local_optimizer.optimize_for_pressure_level(pressure_level)
            
            try:
                result = func(*args, **kwargs)
                return result
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM in {func.__name__}: {e}")
                # 尝试紧急优化
                local_optimizer.optimize_for_pressure_level(MemoryPressureLevel.CRITICAL)
                # 重试一次
                try:
                    result = func(*args, **kwargs)
                    return result
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"CUDA OOM persists after optimization in {func.__name__}")
                    raise
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator

if __name__ == "__main__":
    # 测试内存优化器
    optimizer = MemoryOptimizer()
    
    # 获取当前内存状态
    pressure_level = optimizer.get_memory_pressure_level()
    print(f"Current memory pressure: {pressure_level.value}")
    
    # 获取优化建议
    recommendations = optimizer.get_memory_recommendations()
    print(f"Recommendations: {recommendations}")
    
    # 测试配置优化
    base_config = {
        'batch_size': 32,
        'max_model_len': 4096,
        'max_parallel_seq': 8,
        'max_loras': 5
    }
    
    efficient_config = optimizer.create_memory_efficient_config(base_config)
    print(f"Original config: {base_config}")
    print(f"Efficient config: {efficient_config}")
