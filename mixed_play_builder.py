import os, ray, time, logging, random, traceback, sys
from typing import Dict, List, Optional, Sequence, Union

import unstable
import unstable.reward_transformations as retra
from unstable._types import TrainEnvSpec, EvalEnvSpec
from patch_collector_for_openai import patch_collector_for_openai

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mixed_play_builder")

# 启用详细调试
DEBUG_MODE = True

def debug_log(msg, *args, **kwargs):
    """增强的调试日志函数，包含调用栈信息"""
    if DEBUG_MODE:
        caller_frame = sys._getframe(1)
        caller_info = f"[{caller_frame.f_code.co_name}:{caller_frame.f_lineno}]"
        logger.debug(f"[DEBUG] {caller_info} {msg}", *args, **kwargs)

def log_exception(msg="发生异常"):
    """记录详细的异常信息"""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(f"{msg}:\n{tb_str}")

# 默认配置
_DEFAULT_LORA_CFG = {
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

class MixedPlayEvalEnvSpec(unstable.EvalEnvSpec):
    """增强版EvalEnvSpec，支持不同位置使用不同对手"""
    
    def __init__(self, env_id, num_players, prompt_template, opponent_mapping=None):
        """
        初始化混合对手评估环境规范
        
        Args:
            env_id: 环境ID
            num_players: 环境中的玩家数量
            prompt_template: 使用的提示模板
            opponent_mapping: 玩家位置到对手名称的映射字典
                             例如 {1: "openai-gpt-4o", 2: "openai-gpt-5"}
        """
        super().__init__(env_id=env_id, num_players=num_players, prompt_template=prompt_template)
        self.opponent_mapping = opponent_mapping or {}
        
    def get_opponent_for_position(self, position):
        """获取特定位置的对手名称"""
        return self.opponent_mapping.get(position)

class SelfPlayMixedOpponentSampler(unstable.samplers.model_samplers.BaseModelSampler):
    """结合自对弈和固定对手的模型采样器"""
    
    def __init__(self, model_registry, history_window_size=5, temperature=1.0, 
                 fixed_opponent_weight=0.3, openai_models=None):
        """
        初始化自对弈混合对手采样器
        
        Args:
            model_registry: 模型注册表
            history_window_size: 历史模型窗口大小
            temperature: 采样温度
            fixed_opponent_weight: 固定对手权重
            openai_models: OpenAI模型列表
        """
        super().__init__(model_registry=model_registry)
        self.history_window_size = history_window_size
        self.temperature = temperature
        self.fixed_opponent_weight = fixed_opponent_weight
        self.openai_models = openai_models or []
        self.logger = logging.getLogger("self_play_mixed_sampler")
        self.logger.info(f"初始化自对弈混合对手采样器: 历史窗口={history_window_size}, "
                         f"温度={temperature}, 固定对手权重={fixed_opponent_weight}")
        
    def sample_opponent(self):
        """同步版本的对手采样方法，返回(uid, kind, lora_path, openrouter_name)
        
        注意：此方法是BaseModelSampler的必要实现，用于游戏调度器的训练和评估任务
        """
        self.logger.info("调用sample_opponent方法采样对手")
        try:
            # 获取所有可用模型
            checkpoints = ray.get(self.model_registry.get_all_models.remote())
            
            # 提取checkpoint和fixed模型
            checkpoint_models = {uid: meta for uid, meta in checkpoints.items() if meta.kind == "checkpoint"}
            fixed_models = {uid: meta for uid, meta in checkpoints.items() if meta.kind == "fixed"}
            
            # 记录可用模型数量
            self.logger.debug(f"可用模型: {len(checkpoint_models)}个checkpoint模型, {len(fixed_models)}个固定对手")
            
            selected_uid = None
            selected_kind = None
            selected_path = None
            selected_openrouter_name = None
            
            # 根据配置选择历史模型
            if len(checkpoint_models) > 1:  # 至少有一个历史模型（不包括base）
                # 选择最近的history_window_size个模型
                recent_checkpoints = sorted(
                    [(uid, meta) for uid, meta in checkpoint_models.items()], 
                    key=lambda x: x[1].iteration, 
                    reverse=True
                )
                recent_checkpoints = recent_checkpoints[:self.history_window_size]
                
                # 随机选择是使用历史模型还是固定对手
                if fixed_models and self.fixed_opponent_weight > 0 and random.random() < self.fixed_opponent_weight:
                    # 选择固定对手
                    fixed_uid, fixed_meta = random.choice(list(fixed_models.items()))
                    self.logger.info(f"选择固定对手: {fixed_uid}, 名称: {fixed_meta.path_or_name}")
                    selected_uid = fixed_uid
                    selected_kind = "openrouter"  # 使用openrouter作为kind
                    selected_path = None
                    selected_openrouter_name = fixed_meta.path_or_name
                else:
                    # 选择历史模型
                    checkpoint_uid, checkpoint_meta = random.choice(recent_checkpoints)
                    self.logger.info(f"选择历史模型: {checkpoint_uid} (迭代 {checkpoint_meta.iteration})")
                    selected_uid = checkpoint_uid
                    selected_kind = "checkpoint"
                    selected_path = checkpoint_meta.path_or_name
                    selected_openrouter_name = None
            else:
                # 如果没有历史模型，则使用固定对手
                if fixed_models:
                    fixed_uid, fixed_meta = random.choice(list(fixed_models.items()))
                    self.logger.info(f"选择固定对手: {fixed_uid}, 名称: {fixed_meta.path_or_name}")
                    selected_uid = fixed_uid
                    selected_kind = "openrouter"  # 使用openrouter作为kind
                    selected_path = None
                    selected_openrouter_name = fixed_meta.path_or_name
                else:
                    # 如果没有固定对手，则使用base模型
                    self.logger.info("没有可用的历史模型或固定对手，使用base模型")
                    base_meta = checkpoints.get("base")
                    if not base_meta:
                        raise ValueError("找不到base模型，请确保模型注册表已正确初始化")
                    selected_uid = "base"
                    selected_kind = "checkpoint"
                    selected_path = base_meta.path_or_name
                    selected_openrouter_name = None
            
            self.logger.info(f"采样结果: uid={selected_uid}, kind={selected_kind}, path={selected_path}, openrouter_name={selected_openrouter_name}")
            return selected_uid, selected_kind, selected_path, selected_openrouter_name
        except Exception as e:
            self.logger.error(f"采样对手时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    async def sample(self, *args, **kwargs):
        """采样模型，优先考虑历史模型，然后是固定对手"""
        # 获取所有可用模型
        checkpoints = await self.model_registry.list_checkpoints.remote()
        fixed_opponents = await self.model_registry.list_fixed.remote()
        
        # 根据配置选择历史模型
        if len(checkpoints) > 1:  # 至少有一个历史模型（不包括base）
            # 选择最近的history_window_size个模型
            recent_checkpoints = sorted(checkpoints, key=lambda x: x["iteration"], reverse=True)
            recent_checkpoints = recent_checkpoints[:self.history_window_size]
            
            # 随机选择是使用历史模型还是固定对手
            if fixed_opponents and self.fixed_opponent_weight > 0 and random.random() < self.fixed_opponent_weight:
                # 选择固定对手
                opponent = random.choice(fixed_opponents)
                self.logger.debug(f"选择固定对手: {opponent['name']}")
                return opponent["name"]
            else:
                # 选择历史模型
                checkpoint = random.choice(recent_checkpoints)
                self.logger.debug(f"选择历史模型: {checkpoint['uid']} (迭代 {checkpoint['iteration']})")
                return checkpoint["uid"]
        else:
            # 如果没有历史模型，则使用固定对手
            if fixed_opponents:
                opponent = random.choice(fixed_opponents)
                self.logger.debug(f"选择固定对手: {opponent['name']}")
                return opponent["name"]
            else:
                # 如果没有固定对手，则使用base模型
                self.logger.debug("没有可用的历史模型或固定对手，使用base模型")
                return "base"

def _default_vllm_cfg(model_name: str, lora_cfg: dict, max_generation_len: int) -> dict:
    """创建默认的VLLM配置"""
    return {
        "model_name": model_name,
        "temperature": 0.6,
        "max_tokens": max_generation_len,
        "max_parallel_seq": 32,
        "max_loras": 4,
        "lora_config": lora_cfg,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.8
    }

def _patch_game_scheduler_for_mixed_play(openai_models=None):
    """为混合对弈修补GameScheduler"""
    debug_log("开始修补GameScheduler以支持混合对手评估")
    original_next_eval_job = unstable.GameScheduler.next_eval_job
    
    def patched_next_eval_job(self):
        """修补后的next_eval_job方法，支持混合对手评估"""
        debug_log("调用修补后的next_eval_job")
        # 获取下一个评估环境规范
        try:
            env_spec = self.env_sampler.sample_eval_env()
            debug_log(f"获取到评估环境规范: {env_spec}")
            
            # 检查是否为MixedPlayEvalEnvSpec
            if isinstance(env_spec, MixedPlayEvalEnvSpec) and env_spec.opponent_mapping:
                try:
                    # 为每个位置创建适当的AgentSpec
                    agent_specs = []
                    for pid in range(env_spec.num_players):
                        position = pid + 1  # 位置从1开始
                        
                        # 检查这个位置是否有指定的对手
                        opponent_name = env_spec.get_opponent_for_position(position)
                        
                        if opponent_name and position != env_spec.eval_model_pid:
                            # 为OpenAI对手创建AgentSpec
                            agent_specs.append(unstable.AgentSpec(
                                pid=position,
                                openrouter_name=opponent_name,
                                lora_path=None,
                                prompt_template=env_spec.prompt_template,
                                action_extraction_fn="default",
                                collect_data=False
                            ))
                        else:
                            # 为评估模型创建AgentSpec
                            agent_specs.append(unstable.AgentSpec(
                                pid=position,
                                openrouter_name=None,
                                lora_path="base" if position == env_spec.eval_model_pid else self.model_sampler.sample_sync(),
                                prompt_template=env_spec.prompt_template,
                                action_extraction_fn="default",
                                collect_data=position == env_spec.eval_model_pid
                            ))
                    
                    # 创建GameSpec
                    debug_log(f"为混合对手创建GameSpec，agent_specs={agent_specs}")
                    game_spec = unstable.GameSpec(
                        game_idx=self._eval_game_idx,
                        env_id=env_spec.env_id,
                        seed=random.randint(0, 2**31-1),
                        agent_specs=agent_specs,
                        eval_model_pid=env_spec.eval_model_pid,
                        eval_opponent_name=None  # 不使用单一对手名称
                    )
                    debug_log(f"创建的GameSpec: {game_spec}")
                    return game_spec
                except Exception as e:
                    logger.error(f"创建GameSpec时出错: {e}")
                    log_exception("创建GameSpec异常详情")
                    raise
            else:
                # 回退到原始实现
                debug_log("使用原始next_eval_job实现")
                return original_next_eval_job(self)
        except Exception as e:
            logger.error(f"修补的next_eval_job中出错: {e}")
            log_exception("next_eval_job异常详情")
            raise
    
    # 应用修补
    try:
        unstable.GameScheduler.next_eval_job = patched_next_eval_job
        debug_log("成功替换GameScheduler.next_eval_job方法")
        logger.info("已修补GameScheduler以支持混合对手评估")
    except Exception as e:
        logger.error(f"修补GameScheduler时出错: {e}")
        log_exception("修补GameScheduler异常详情")
        raise

class _MixedPlayRun:
    """混合对弈运行类"""
    
    def __init__(self, *, collector, learner):
        debug_log(f"初始化_MixedPlayRun，collector={collector}, learner={learner}")
        self.collector, self.learner = collector, learner
        
    def start(self, learning_steps: int = 200, num_collection_workers: int = 64, num_eval_workers: int = 8):
        """开始训练"""
        debug_log(f"开始训练，learning_steps={learning_steps}, num_collection_workers={num_collection_workers}, num_eval_workers={num_eval_workers}")
        try:
            debug_log("启动收集器...")
            collect_ref = self.collector.collect.remote(num_train_workers=num_collection_workers, num_eval_workers=num_eval_workers)
            debug_log(f"收集器启动成功，引用ID: {collect_ref}")
            
            debug_log("启动学习器...")
            train_ref = self.learner.train.remote(learning_steps)
            debug_log(f"学习器启动成功，引用ID: {train_ref}")
            
            debug_log("等待学习器完成训练...")
            try:
                ray.get(train_ref)
                debug_log("学习器训练完成")
            except Exception as e:
                logger.error(f"学习器训练过程中出错: {e}")
                log_exception("学习器异常详情")
                raise
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            log_exception("训练异常详情")
            raise
        finally:
            debug_log("清理资源，终止收集器...")
            try:
                ray.kill(self.collector, no_restart=True)
                debug_log("收集器已终止")
            except Exception as e:
                logger.error(f"终止收集器时出错: {e}")
                log_exception("终止收集器异常详情")

def build_mixed_play(*, 
                    model_name: str,
                    train_envs: Sequence[TrainEnvSpec],
                    eval_envs: Sequence[MixedPlayEvalEnvSpec],
                    openai_config: Dict,
                    fixed_opponents: List[str],
                    algorithm: str = "a2c",
                    max_train_len: Optional[int] = None,
                    max_generation_len: int = 4096,
                    batch_size: int = 16,
                    mini_batch_size: int = 1,
                    learning_rate: float = 1e-5,
                    gradient_clipping: float = 0.2,
                    buffer_size: Optional[int] = None,
                    lora_config: Optional[dict] = None,
                    vllm_config: Optional[dict] = None,
                    history_window_size: int = 5,
                    fixed_opponent_weight: float = 0.3,
                    wandb_project: str = "UnstableBaselines"):
    """
    构建混合对弈训练系统
    
    Args:
        model_name: 模型名称
        train_envs: 训练环境规范列表
        eval_envs: 评估环境规范列表
        openai_config: OpenAI配置
        fixed_opponents: 固定对手列表
        algorithm: 算法名称，默认为"a2c"
        max_train_len: 最大训练长度
        max_generation_len: 最大生成长度
        batch_size: 批量大小
        mini_batch_size: 小批量大小
        learning_rate: 学习率
        gradient_clipping: 梯度裁剪
        buffer_size: 缓冲区大小
        lora_config: LoRA配置
        vllm_config: VLLM配置
        history_window_size: 历史窗口大小
        fixed_opponent_weight: 固定对手权重
        wandb_project: Wandb项目名称
    
    Returns:
        _MixedPlayRun: 混合对弈运行实例
    """
    # 应用OpenAI代理补丁
    logger.info(f"应用OpenAI代理补丁，配置: {openai_config}")
    try:
        debug_log("调用patch_collector_for_openai...")
        patch_collector_for_openai(openai_config)
        debug_log("OpenAI代理补丁应用成功")
    except Exception as e:
        logger.error(f"应用OpenAI代理补丁时出错: {e}")
        log_exception("应用OpenAI代理补丁异常详情")
        raise
    
    # 初始化Ray（如果尚未初始化）
    logger.info("初始化Ray...")
    try:
        if not ray.is_initialized():
            logger.info("Ray未初始化，现在初始化...")
            debug_log("调用ray.init，自动检测GPU资源（不手动指定num_gpus）")
            ray.init(namespace="unstable")
            debug_log(f"Ray初始化成功，资源: {ray.available_resources()}")
        else:
            logger.info("Ray已初始化，跳过初始化")
            debug_log(f"Ray已经初始化，详细资源情况: {ray.available_resources()}")
            debug_log(f"Ray内部状态: 命名空间={ray.get_runtime_context().namespace}")
            
            # 检查GPU资源是否正确分配
            resources = ray.available_resources()
            if 'GPU' not in resources or resources['GPU'] < 1:
                logger.warning(f"[GPU资源警告] Ray中没有可用的GPU资源! 资源详情: {resources}")
                debug_log("尝试获取更多关于Ray集群的信息...")
                try:
                    cluster_resources = ray.cluster_resources()
                    debug_log(f"Ray集群总资源: {cluster_resources}")
                except Exception as e:
                    debug_log(f"获取集群资源时出错: {e}")
    except Exception as e:
        logger.error(f"检查或初始化Ray时出错: {e}")
        log_exception("初始化Ray异常详情")
        raise
    
    # 环境采样器
    logger.info("配置环境采样器...")
    env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
        train_env_specs=train_envs,
        eval_env_specs=eval_envs
    )
    logger.info(f"环境采样器配置成功，环境: {env_sampler.env_list()}")
    
    # 跟踪器
    logger.info("初始化跟踪器...")
    tracker = unstable.Tracker.options(name="Tracker").remote(
        run_name=f"MixedPlay-{model_name.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}",
        wandb_project=wandb_project
    )
    logger.info("跟踪器初始化成功!")
    
    # 初始化模型注册表
    logger.info("初始化模型注册表...")
    model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
    
    # 添加基础检查点，确保路径有效
    base_checkpoint_path = f"checkpoints/base_{int(time.time())}"
    logger.info(f"添加基础检查点: uid=base, path={base_checkpoint_path}")
    try:
        ray.get(model_registry.add_checkpoint.remote(uid="base", path=base_checkpoint_path, iteration=0))
        logger.info("基础检查点添加成功")
    except Exception as e:
        logger.error(f"添加基础检查点失败: {e}")
        log_exception("添加基础检查点异常详情")
        raise
    
    # 添加固定对手
    for opponent_name in fixed_opponents:
        logger.info(f"添加固定对手: {opponent_name}")
        ray.get(model_registry.add_fixed.remote(name=opponent_name))
    logger.info("模型注册表初始化成功!")
    
    # 初始化对手采样器
    logger.info("初始化模型采样器...")
    model_sampler = SelfPlayMixedOpponentSampler(
        model_registry=model_registry,
        history_window_size=history_window_size,
        fixed_opponent_weight=fixed_opponent_weight
    )
    logger.info("模型采样器初始化成功!")
    
    # 为混合对弈修补GameScheduler
    _patch_game_scheduler_for_mixed_play()
    
    # 构建游戏调度器
    logger.info("构建游戏调度器...")
    game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(
        model_sampler=model_sampler,
        env_sampler=env_sampler,
        logging_dir=ray.get(tracker.get_log_dir.remote())
    )
    logger.info("游戏调度器构建成功!")
    
    # 构建缓冲区
    logger.info("初始化数据缓冲区...")
    buffer_size = buffer_size or batch_size * 2
    
    # 根据算法选择适当的缓冲区类型
    if algorithm == "reinforce":
        buffer = unstable.StepBuffer.options(name="Buffer").remote(
            max_buffer_size=buffer_size,
            tracker=tracker,
            final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
            step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
            sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)])
        )
    elif algorithm == "a2c":
        buffer = unstable.EpisodeBuffer.options(name="Buffer").remote(
            max_buffer_size=buffer_size,
            tracker=tracker,
            final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
            step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
            sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)])
        )
    else:
        raise NotImplementedError(f"算法 {algorithm} 未分配到特定的缓冲区类型。")
    logger.info("数据缓冲区初始化成功!")
    
    # 初始化收集器
    logger.info("初始化收集器...")
    _lora_cfg = lora_config or _DEFAULT_LORA_CFG
    
    # 准备VLLM配置
    vllm_cfg = vllm_config or _default_vllm_cfg(model_name, _lora_cfg, max_generation_len)
    debug_log(f"VLLM配置: {vllm_cfg}")
    
    # 检查Ray资源状态
    debug_log(f"创建Collector前的Ray资源: {ray.available_resources()}")
    
    try:
        debug_log("开始创建Collector...")
        # 获取Ray资源信息
        resources = ray.available_resources()
        debug_log(f"详细的Ray资源信息: {resources}")
        
        # 检查是否有可用的GPU
        has_gpu = 'GPU' in resources and resources['GPU'] > 0
        debug_log(f"系统中有可用的GPU: {has_gpu}")
        
        # 准备额外的资源请求
        custom_resources = {}
        # 如果有特殊的GPU类型，添加到自定义资源
        for key in resources:
            if key != 'GPU' and key != 'CPU' and 'GPU' in key:
                custom_resources[key] = 1
                debug_log(f"添加特殊的GPU资源请求: {key}")
        
        debug_log(f"使用num_gpus=1和额外资源: {custom_resources}")
        
        # 正确使用num_cpus和num_gpus参数
        collector = unstable.Collector.options(
            name="Collector",
            num_cpus=1,
            num_gpus=1,
            resources=custom_resources  # 只包含特殊的GPU类型资源
        ).remote(
            vllm_config=vllm_cfg,
            tracker=tracker,
            buffer=buffer,
            game_scheduler=game_scheduler
        )
        debug_log(f"Collector创建成功: {collector}")
        debug_log(f"Collector创建后的Ray资源: {ray.available_resources()}")
    except Exception as e:
        logger.error(f"创建Collector时出错: {e}")
        log_exception("创建Collector异常详情")
        raise
    
    logger.info("收集器初始化成功!")
    
    # 初始化学习器
    logger.info("初始化学习器...")
    
    # 检查可用的GPU资源
    resources = ray.available_resources()
    debug_log(f"学习器初始化前的Ray资源: {resources}")
    
    # 准备学习器的资源配置
    try:
        # 检查是否有可用的GPU
        has_gpu = 'GPU' in resources and resources['GPU'] > 0
        debug_log(f"学习器初始化时系统中有可用的GPU: {has_gpu}")
        
        # 准备额外的资源请求
        custom_resources = {}
        # 如果有特殊的GPU类型，添加到自定义资源
        for key in resources:
            if key != 'GPU' and key != 'CPU' and 'GPU' in key:
                custom_resources[key] = 1
                debug_log(f"添加学习器特殊的GPU资源请求: {key}")
        
        debug_log(f"学习器使用num_gpus=1和额外资源: {custom_resources}")
        
        # 让 Ray 自动为 Actor 分配并设置 CUDA_VISIBLE_DEVICES，不要手动覆盖，以避免 invalid device ordinal
        
        if algorithm == "reinforce":
            learner = unstable.REINFORCELearner.options(
                name="Learner",
                num_cpus=1,
                num_gpus=1,
                resources=custom_resources
            ).remote(
                model_name=model_name,
                lora_cfg=_lora_cfg,
                batch_size=batch_size,
                mini_batch_size=mini_batch_size,
                learning_rate=learning_rate,
                grad_clip=gradient_clipping,
                buffer=buffer,
                tracker=tracker,
                model_registry=model_registry
            )
        elif algorithm == "a2c":
            learner = unstable.A2CLearner.options(
                name="Learner",
                num_cpus=1,
                num_gpus=1,
                resources=custom_resources
            ).remote(
                model_name=model_name,
                lora_cfg=_lora_cfg,
                batch_size=batch_size,
                mini_batch_size=mini_batch_size,
                learning_rate=learning_rate,
                grad_clip=gradient_clipping,
                buffer=buffer,
                tracker=tracker,
                model_registry=model_registry
            )
        else:
            raise NotImplementedError(f"不支持的算法: {algorithm}")
        
        debug_log(f"学习器创建成功: {learner}")
        debug_log(f"学习器创建后的Ray资源: {ray.available_resources()}")
    except Exception as e:
        logger.error(f"创建学习器时出错: {e}")
        log_exception("创建学习器异常详情")
        raise
        
    logger.info("学习器初始化成功!")
    
    # 初始化算法
    logger.info("初始化算法...")
    if algorithm == "reinforce":
        ray.get(learner.initialize_algorithm.remote(
            max_train_len=max_train_len,
            max_generation_len=max_generation_len
        ))
    elif algorithm == "a2c":
        ray.get(learner.initialize_algorithm.remote(
            infer_mini_batch_size=16,
            critic_learning_rate=5e-5,
            normalize_adv=True,
            max_train_len=max_train_len,
            max_generation_len=max_generation_len
        ))
    logger.info("算法初始化成功!")
    
    return _MixedPlayRun(collector=collector, learner=learner)
