#!/usr/bin/env python
"""
自对弈+混合对手训练系统

这个脚本实现了一个高级训练系统，结合了以下特性：
1. 自对弈：模型与自己的历史快照对战
2. 混合对手：同时与多个外部模型（如OpenAI模型）对战
3. 均衡训练：通过动态采样确保模型达到纳什均衡

这种训练方式类似于AlphaGo的训练方法，但增加了外部模型作为对手，
使得训练更加多样化，避免过拟合于自身策略。
"""

import os
import sys
import ray
import time
import random
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# 导入必要的模块
import unstable
from unstable._types import AgentSpec, GameSpec
from patch_collector_for_openai import patch_collector_for_openai

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('self_play_mixed_opponents.log')
    ]
)
logger = logging.getLogger("self_play_mixed")

# 训练配置
ENV_ID = "ThreePlayerIPD-v0"  # 环境ID
MODEL_NAME = "qwen/Qwen1.5-7B"  # 基础模型名称
NUM_ITERATIONS = 1000  # 训练迭代次数
SAVE_INTERVAL = 10  # 模型保存间隔
EVAL_INTERVAL = 5  # 评估间隔
HISTORY_WINDOW_SIZE = 20  # 历史模型窗口大小
HISTORY_SAMPLING_TEMP = 1.0  # 历史模型采样温度

# OpenAI模型配置
OPENAI_MODELS = {
    "gpt4o": "gpt-4o",
    "gpt4o_mini": "gpt-4o-mini",
    "gpt5": "gpt-5",
    "gpt5_chat": "gpt-5-chat"
}

# 创建opponent名称（添加openai-前缀）
OPPONENT_NAMES = {key: f"openai-{value}" for key, value in OPENAI_MODELS.items()}

# OpenAI配置
openai_config = {
    "model_name": "gpt-4o",  # 默认模型
    "verbose": True,
}

# 采样权重配置
SAMPLING_WEIGHTS = {
    "self_play": 0.6,  # 自对弈权重
    "external": 0.4,   # 外部模型权重
}

# 自定义模型采样器，支持历史快照和外部模型混合采样
class SelfPlayMixedOpponentSampler(unstable.samplers.model_samplers.BaseModelSampler):
    """
    自对弈+混合对手模型采样器
    
    支持从以下来源采样对手：
    1. 历史模型快照（自对弈）
    2. 固定外部模型（如OpenAI模型）
    
    采样策略可以通过权重配置进行调整
    """
    
    def __init__(
        self, 
        model_registry, 
        history_window_size=20, 
        history_sampling_temp=1.0,
        sampling_weights=None
    ):
        super().__init__(model_registry=model_registry)
        self.history_window_size = history_window_size
        self.history_sampling_temp = history_sampling_temp
        self.sampling_weights = sampling_weights or {"self_play": 0.7, "external": 0.3}
        self.logger = logging.getLogger("SelfPlayMixedSampler")
        
        # 验证权重总和为1
        total_weight = sum(self.sampling_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            self.logger.warning(f"Sampling weights do not sum to 1.0: {total_weight}. Normalizing...")
            for k in self.sampling_weights:
                self.sampling_weights[k] /= total_weight
    
    def sample_opponent(self):
        """
        采样一个对手
        
        根据配置的权重，决定是从历史快照中采样还是从外部模型中采样
        """
        # 决定采样来源
        source = self._sample_source()
        
        if source == "self_play":
            # 从历史快照中采样
            return self._sample_from_history()
        else:
            # 从固定外部模型中采样
            return self._sample_from_external()
    
    def _sample_source(self):
        """根据权重决定采样来源"""
        sources = list(self.sampling_weights.keys())
        weights = list(self.sampling_weights.values())
        return random.choices(sources, weights=weights, k=1)[0]
    
    def _sample_from_history(self):
        """从历史快照中采样"""
        # 获取所有模型
        all_models = ray.get(self.model_registry.get_all_models.remote())
        
        # 过滤出checkpoint类型的模型（历史快照）
        checkpoints = {
            uid: meta for uid, meta in all_models.items() 
            if meta.kind == "checkpoint" and uid != "base"
        }
        
        if not checkpoints:
            self.logger.warning("No historical checkpoints available, falling back to base model")
            return "base", "checkpoint", None, None
        
        # 按照迭代次数排序
        sorted_checkpoints = sorted(
            checkpoints.items(), 
            key=lambda x: x[1].iteration if x[1].iteration is not None else 0
        )
        
        # 只保留最近的history_window_size个快照
        recent_checkpoints = sorted_checkpoints[-self.history_window_size:]
        
        if not recent_checkpoints:
            self.logger.warning("No recent checkpoints available, falling back to base model")
            return "base", "checkpoint", None, None
        
        # 计算采样权重（使用softmax温度）
        iterations = np.array([meta.iteration for _, meta in recent_checkpoints])
        if self.history_sampling_temp > 0:
            weights = np.exp(iterations / self.history_sampling_temp)
            weights = weights / np.sum(weights)
        else:
            # 均匀采样
            weights = np.ones_like(iterations) / len(iterations)
        
        # 采样一个checkpoint
        idx = np.random.choice(len(recent_checkpoints), p=weights)
        uid, meta = recent_checkpoints[idx]
        
        self.logger.info(f"Sampled historical checkpoint: {uid} (iteration {meta.iteration})")
        return uid, "checkpoint", meta.path_or_name, None
    
    def _sample_from_external(self):
        """从固定外部模型中采样"""
        # 获取所有模型
        all_models = ray.get(self.model_registry.get_all_models.remote())
        
        # 过滤出固定模型（外部模型）
        fixed_models = {
            uid: meta for uid, meta in all_models.items() 
            if meta.kind == "openrouter" and meta.path_or_name.startswith("openai-")
        }
        
        if not fixed_models:
            self.logger.warning("No external models available, falling back to self-play")
            return self._sample_from_history()
        
        # 随机选择一个外部模型
        uid = random.choice(list(fixed_models.keys()))
        meta = fixed_models[uid]
        
        self.logger.info(f"Sampled external model: {uid} ({meta.path_or_name})")
        return uid, meta.kind, None, meta.path_or_name

# 自定义游戏调度器，支持位置特定的对手分配
class MixedPlayEvalGameScheduler:
    """
    混合对手评估游戏调度器
    
    允许为特定位置分配特定的对手，用于评估模型在不同对手组合下的表现
    """
    
    def __init__(self, original_scheduler):
        self.original_scheduler = original_scheduler
        self.position_opponents = {}  # 位置特定的对手映射
        self.logger = logging.getLogger("MixedPlayEvalScheduler")
    
    def set_position_opponent(self, position, opponent_name):
        """为特定位置设置特定的对手"""
        self.position_opponents[position] = opponent_name
        self.logger.info(f"Set position {position} to use opponent: {opponent_name}")
    
    def next_eval_job(self):
        """获取下一个评估任务，应用位置特定的对手分配"""
        # 获取原始评估任务
        job = self.original_scheduler.next_eval_job()
        
        if job is None:
            return None
        
        # 修改agent_specs，应用位置特定的对手
        new_agent_specs = []
        for agent_spec in job.agent_specs:
            if agent_spec.pid in self.position_opponents:
                # 为特定位置分配特定对手
                opponent_name = self.position_opponents[agent_spec.pid]
                self.logger.info(f"Using opponent {opponent_name} for position {agent_spec.pid}")
                
                # 创建新的agent_spec
                new_spec = AgentSpec(
                    pid=agent_spec.pid,
                    kind="openrouter",  # 使用openrouter作为kind
                    collect_data=agent_spec.collect_data,
                    openrouter_name=opponent_name,
                    prompt_template=agent_spec.prompt_template,
                    action_extraction_fn=agent_spec.action_extraction_fn
                )
                new_agent_specs.append(new_spec)
            else:
                # 保持原样
                new_agent_specs.append(agent_spec)
        
        # 创建新的GameSpec
        new_job = GameSpec(
            game_idx=job.game_idx,
            env_id=job.env_id,
            seed=job.seed,
            agent_specs=new_agent_specs,
            eval_model_pid=job.eval_model_pid,
            eval_opponent_name=job.eval_opponent_name
        )
        
        return new_job

# 主函数
def main():
    """主函数"""
    logger.info("启动自对弈+混合对手训练系统")
    
    # 检查OpenAI API密钥
    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY环境变量未设置，无法使用OpenAI模型")
        return
    
    # 应用OpenAI代理补丁
    logger.info("应用OpenAI代理补丁...")
    try:
        patch_collector_for_openai(openai_config)
        logger.info("补丁应用成功！")
    except Exception as e:
        logger.error(f"应用补丁失败: {e}", exc_info=True)
        return
    
    # 初始化Ray
    logger.info("初始化Ray...")
    try:
        ray.init(
            namespace="unstable",
            num_gpus=4,  # Specify 4 A100 GPUs
            _memory=2**33,  # 8GB memory limit
            object_store_memory=2**33,  # 8GB object store memory
            object_spilling_directory="/tmp/ray_spill"
        )
        logger.info("Ray initialized successfully!")
    except Exception as e:
        logger.error(f"Ray初始化失败: {e}", exc_info=True)
        return
    
    # 配置环境采样器
    logger.info("配置环境采样器...")
    try:
        from unstable._types import TrainEnvSpec
        # 创建训练环境规范
        train_env_specs = [
            TrainEnvSpec(
                env_id=ENV_ID,
                num_envs=1,
                seed=42,
                options={}
            )
        ]
        # 创建评估环境规范（与训练环境相同）
        eval_env_specs = [
            TrainEnvSpec(
                env_id=ENV_ID,
                num_envs=1,
                seed=4242,
                options={}
            )
        ]
        env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
            train_env_specs=train_env_specs,
            eval_env_specs=eval_env_specs
        )
        logger.info(f"环境采样器配置成功，环境: {env_sampler.env_list()}")
    except Exception as e:
        logger.error(f"配置环境采样器失败: {e}", exc_info=True)
        return
    
    # 初始化跟踪器
    logger.info("初始化跟踪器...")
    try:
        tracker = unstable.Tracker.options(name="Tracker").remote(
            run_name=f"SelfPlayMixed-{MODEL_NAME.split('/')[-1]}-{ENV_ID}-{int(time.time())}", 
            wandb_project="UnstableBaselines"
        )
        logger.info("跟踪器初始化成功！")
    except Exception as e:
        logger.error(f"初始化跟踪器失败: {e}", exc_info=True)
        return
    
    # 初始化模型注册表
    logger.info("初始化模型注册表...")
    try:
        model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
        ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
        
        # 添加所有OpenAI代理作为固定对手
        for name, opponent_name in OPPONENT_NAMES.items():
            logger.info(f"添加固定对手: {opponent_name}")
            ray.get(model_registry.add_fixed.remote(name=opponent_name))
        
        logger.info("模型注册表初始化成功！")
    except Exception as e:
        logger.error(f"初始化模型注册表失败: {e}", exc_info=True)
        return
    
    # 初始化自定义模型采样器
    logger.info("初始化自定义模型采样器...")
    try:
        model_sampler = SelfPlayMixedOpponentSampler(
            model_registry=model_registry,
            history_window_size=HISTORY_WINDOW_SIZE,
            history_sampling_temp=HISTORY_SAMPLING_TEMP,
            sampling_weights=SAMPLING_WEIGHTS
        )
        logger.info("自定义模型采样器初始化成功！")
    except Exception as e:
        logger.error(f"初始化自定义模型采样器失败: {e}", exc_info=True)
        return
    
    # 初始化游戏调度器
    logger.info("初始化游戏调度器...")
    try:
        game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(
            env_sampler=env_sampler,
            model_sampler=model_sampler,
            n_eval_games=5,
            eval_freq=EVAL_INTERVAL
        )
        logger.info("游戏调度器初始化成功！")
        
        # 创建混合对手评估调度器
        eval_scheduler = MixedPlayEvalGameScheduler(game_scheduler)
        
        # 为特定位置分配特定对手（用于评估）
        eval_scheduler.set_position_opponent(1, OPPONENT_NAMES["gpt4o"])
        eval_scheduler.set_position_opponent(2, OPPONENT_NAMES["gpt5"])
        
        # 猴子补丁游戏调度器的next_eval_job方法
        game_scheduler.next_eval_job = eval_scheduler.next_eval_job
        logger.info("混合对手评估调度器配置成功！")
    except Exception as e:
        logger.error(f"初始化游戏调度器失败: {e}", exc_info=True)
        return
    
    # 初始化缓冲区
    logger.info("初始化缓冲区...")
    try:
        buffer = unstable.Buffer.options(name="Buffer").remote(
            tracker=tracker,
            capacity=10000,
            sample_batch_size=256
        )
        logger.info("缓冲区初始化成功！")
    except Exception as e:
        logger.error(f"初始化缓冲区失败: {e}", exc_info=True)
        return
    
    # 初始化收集器
    logger.info("初始化收集器...")
    try:
        collector = unstable.Collector.options(name="Collector").remote(
            tracker=tracker,
            model_registry=model_registry,
            game_scheduler=game_scheduler,
            buffer=buffer,
            n_actors=4,
            actor_cls=unstable.Actor,
            base_model=MODEL_NAME,
            prompt_template="qwen3-no-reasoning",
            action_extraction_fn="default"
        )
        logger.info("收集器初始化成功！")
    except Exception as e:
        logger.error(f"初始化收集器失败: {e}", exc_info=True)
        return
    
    # 初始化学习器
    logger.info("初始化学习器...")
    try:
        learner = unstable.Learner.options(name="Learner").remote(
            tracker=tracker,
            model_registry=model_registry,
            buffer=buffer,
            base_model=MODEL_NAME,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            learning_rate=1e-4,
            save_freq=SAVE_INTERVAL
        )
        logger.info("学习器初始化成功！")
    except Exception as e:
        logger.error(f"初始化学习器失败: {e}", exc_info=True)
        return
    
    # 开始训练
    logger.info(f"开始训练，计划迭代次数: {NUM_ITERATIONS}")
    try:
        ray.get(collector.run.remote())
        ray.get(learner.run.remote(NUM_ITERATIONS))
        logger.info("训练完成！")
    except Exception as e:
        logger.error(f"训练过程中出错: {e}", exc_info=True)
    finally:
        # 关闭Ray
        ray.shutdown()
        logger.info("Ray已关闭")

if __name__ == "__main__":
    main()
