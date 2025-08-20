"""
GPU监控工具，用于跟踪各个模块的GPU内存使用情况
"""

import time
import threading
import psutil
import GPUtil
import torch
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
import os
from datetime import datetime

@dataclass
class GPUSnapshot:
    """GPU状态快照"""
    timestamp: float
    gpu_id: int
    memory_used: float  # MB
    memory_total: float  # MB
    memory_percent: float
    temperature: float
    utilization: float
    power_draw: float
    module_name: str = "unknown"

@dataclass
class GPUStats:
    """GPU统计信息"""
    peak_memory: float = 0.0
    avg_memory: float = 0.0
    min_memory: float = float('inf')
    total_samples: int = 0
    snapshots: List[GPUSnapshot] = field(default_factory=list)
    
    def update(self, snapshot: GPUSnapshot):
        self.snapshots.append(snapshot)
        self.peak_memory = max(self.peak_memory, snapshot.memory_used)
        self.min_memory = min(self.min_memory, snapshot.memory_used)
        self.total_samples += 1
        self.avg_memory = sum(s.memory_used for s in self.snapshots) / len(self.snapshots)

class GPUMonitor:
    """GPU监控器"""
    
    def __init__(self, 
                 sample_interval: float = 0.5,
                 max_history: int = 1000,
                 log_file: Optional[str] = None):
        self.sample_interval = sample_interval
        self.max_history = max_history
        self.log_file = log_file
        
        # 监控数据
        self.stats_by_module: Dict[str, GPUStats] = defaultdict(GPUStats)
        self.current_module = "unknown"
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 日志设置
        self.logger = logging.getLogger("gpu_monitor")
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def set_current_module(self, module_name: str):
        """设置当前监控的模块名称"""
        self.current_module = module_name
        self.logger.info(f"Switched to monitoring module: {module_name}")
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                self._collect_gpu_stats()
                time.sleep(self.sample_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _collect_gpu_stats(self):
        """收集GPU统计信息"""
        try:
            # 使用GPUtil获取GPU信息
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                snapshot = GPUSnapshot(
                    timestamp=time.time(),
                    gpu_id=gpu.id,
                    memory_used=gpu.memoryUsed,
                    memory_total=gpu.memoryTotal,
                    memory_percent=gpu.memoryUtil * 100,
                    temperature=gpu.temperature,
                    utilization=gpu.load * 100,
                    power_draw=getattr(gpu, 'powerDraw', 0.0),
                    module_name=self.current_module
                )
                
                # 更新统计信息
                self.stats_by_module[self.current_module].update(snapshot)
                
                # 限制历史记录长度
                if len(self.stats_by_module[self.current_module].snapshots) > self.max_history:
                    self.stats_by_module[self.current_module].snapshots.pop(0)
                
                # 记录高内存使用情况
                if snapshot.memory_percent > 80:
                    self.logger.warning(
                        f"High GPU memory usage: {snapshot.memory_percent:.1f}% "
                        f"({snapshot.memory_used:.0f}MB/{snapshot.memory_total:.0f}MB) "
                        f"in module {self.current_module}"
                    )
        
        except Exception as e:
            self.logger.error(f"Failed to collect GPU stats: {e}")
    
    def get_stats_summary(self) -> Dict[str, Dict]:
        """获取统计摘要"""
        summary = {}
        for module_name, stats in self.stats_by_module.items():
            if stats.total_samples > 0:
                summary[module_name] = {
                    "peak_memory_mb": stats.peak_memory,
                    "avg_memory_mb": stats.avg_memory,
                    "min_memory_mb": stats.min_memory,
                    "peak_memory_percent": (stats.peak_memory / stats.snapshots[-1].memory_total * 100) if stats.snapshots else 0,
                    "total_samples": stats.total_samples,
                    "duration_seconds": (stats.snapshots[-1].timestamp - stats.snapshots[0].timestamp) if len(stats.snapshots) > 1 else 0
                }
        return summary
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_stats_summary()
        print("\n" + "="*80)
        print("GPU MEMORY USAGE SUMMARY")
        print("="*80)
        
        for module_name, stats in summary.items():
            print(f"\nModule: {module_name}")
            print(f"  Peak Memory: {stats['peak_memory_mb']:.1f} MB ({stats['peak_memory_percent']:.1f}%)")
            print(f"  Avg Memory:  {stats['avg_memory_mb']:.1f} MB")
            print(f"  Min Memory:  {stats['min_memory_mb']:.1f} MB")
            print(f"  Duration:    {stats['duration_seconds']:.1f} seconds")
            print(f"  Samples:     {stats['total_samples']}")
    
    def save_detailed_report(self, filepath: str):
        """保存详细报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_stats_summary(),
            "detailed_snapshots": {}
        }
        
        for module_name, stats in self.stats_by_module.items():
            report["detailed_snapshots"][module_name] = [
                {
                    "timestamp": s.timestamp,
                    "gpu_id": s.gpu_id,
                    "memory_used": s.memory_used,
                    "memory_percent": s.memory_percent,
                    "temperature": s.temperature,
                    "utilization": s.utilization,
                    "power_draw": s.power_draw
                }
                for s in stats.snapshots
            ]
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed GPU report saved to: {filepath}")

class GPUMemoryManager:
    """GPU内存管理器，用于缓解内存压力"""
    
    def __init__(self):
        self.logger = logging.getLogger("gpu_memory_manager")
    
    def clear_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("GPU cache cleared")
    
    def get_memory_info(self) -> Dict[int, Dict]:
        """获取所有GPU的内存信息"""
        memory_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    memory_info[i] = {
                        "allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                        "cached": torch.cuda.memory_reserved() / 1024**2,      # MB
                        "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,  # MB
                        "max_cached": torch.cuda.max_memory_reserved() / 1024**2       # MB
                    }
        return memory_info
    
    def optimize_memory_usage(self):
        """优化内存使用"""
        self.clear_cache()
        
        # 重置峰值内存统计
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(device=i)
                except RuntimeError as e:
                    self.logger.warning(f"Could not reset peak memory stats for GPU {i}: {e}")
        
        self.logger.info("Memory optimization completed")
    
    def check_memory_pressure(self, threshold_percent: float = 85.0) -> List[int]:
        """检查内存压力，返回高内存使用的GPU ID列表"""
        high_usage_gpus = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                if gpu.memoryUtil * 100 > threshold_percent:
                    high_usage_gpus.append(gpu.id)
                    self.logger.warning(
                        f"GPU {gpu.id} memory usage: {gpu.memoryUtil*100:.1f}% "
                        f"({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)"
                    )
        except Exception as e:
            self.logger.error(f"Failed to check memory pressure: {e}")
        
        return high_usage_gpus

# 装饰器：用于自动监控函数的GPU使用情况
def gpu_monitor(module_name: str, monitor_instance: Optional[GPUMonitor] = None):
    """装饰器：监控函数的GPU使用情况"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            if monitor_instance:
                monitor_instance.set_current_module(module_name)
            
            # 记录开始时的内存状态
            start_memory = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    start_memory[i] = torch.cuda.memory_allocated(i)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 记录结束时的内存状态
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        end_memory = torch.cuda.memory_allocated(i)
                        delta = (end_memory - start_memory.get(i, 0)) / 1024**2  # MB
                        if abs(delta) > 10:  # 只记录显著变化
                            logging.getLogger("gpu_monitor").info(
                                f"GPU {i} memory change in {module_name}: {delta:+.1f} MB"
                            )
        
        return wrapper
    return decorator

# 上下文管理器：用于监控代码块的GPU使用情况
class gpu_monitor_context:
    """上下文管理器：监控代码块的GPU使用情况"""
    
    def __init__(self, module_name: str, monitor_instance: Optional[GPUMonitor] = None):
        self.module_name = module_name
        self.monitor = monitor_instance
        self.start_memory = {}
    
    def __enter__(self):
        if self.monitor:
            self.monitor.set_current_module(self.module_name)
        
        # 记录开始时的内存状态
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.start_memory[i] = torch.cuda.memory_allocated(i)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 记录结束时的内存状态
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                end_memory = torch.cuda.memory_allocated(i)
                delta = (end_memory - self.start_memory.get(i, 0)) / 1024**2  # MB
                if abs(delta) > 10:  # 只记录显著变化
                    logging.getLogger("gpu_monitor").info(
                        f"GPU {i} memory change in {self.module_name}: {delta:+.1f} MB"
                    )

if __name__ == "__main__":
    # 测试GPU监控器
    monitor = GPUMonitor(sample_interval=0.1, log_file="gpu_monitor_test.log")
    memory_manager = GPUMemoryManager()
    
    print("Starting GPU monitoring test...")
    monitor.start_monitoring()
    
    try:
        # 模拟不同模块的GPU使用
        monitor.set_current_module("test_module_1")
        time.sleep(2)
        
        monitor.set_current_module("test_module_2")
        time.sleep(2)
        
        # 检查内存压力
        high_usage = memory_manager.check_memory_pressure()
        if high_usage:
            print(f"High memory usage detected on GPUs: {high_usage}")
            memory_manager.optimize_memory_usage()
        
    finally:
        monitor.stop_monitoring()
        monitor.print_summary()
        monitor.save_detailed_report("gpu_monitor_report.json")
