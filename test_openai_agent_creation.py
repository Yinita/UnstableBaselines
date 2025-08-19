#!/usr/bin/env python
"""
测试脚本：验证OpenAI固定对手注册和agent创建
"""

import os
import sys
import ray
import time
import logging
import traceback
from typing import Dict, Any

# 导入必要的模块
import unstable
from defined_agents import OpenAIAgent
from patch_collector_for_openai import create_openai_agent_for_opponent, patch_collector_for_openai

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_test_debug.log')
    ]
)
logger = logging.getLogger("agent_test")

# 测试配置
OPPONENT_MODELS = {
    "gpt4o": "gpt-4o",
    "gpt4o_mini": "gpt-4o-mini",
    "gpt5": "gpt-5",
    "gpt5_chat": "gpt-5-chat"
}

# 创建opponent名称（添加openai-前缀）
OPPONENT_NAMES = {key: f"openai-{value}" for key, value in OPPONENT_MODELS.items()}

# OpenAI配置
openai_config = {
    "model_name": "gpt-4o",  # 默认模型
    "verbose": True,
}

def test_direct_agent_creation():
    """直接测试OpenAIAgent创建"""
    logger.info("=== 测试直接创建OpenAIAgent ===")
    
    for name, model in OPPONENT_MODELS.items():
        logger.info(f"尝试创建模型: {model}")
        try:
            # 直接创建OpenAIAgent
            agent = OpenAIAgent(model_name=model, verbose=True)
            logger.info(f"✅ 成功创建 {model} 代理")
            
            # 测试简单请求
            logger.info(f"测试 {model} 的简单请求...")
            response = agent.__call__("Hello, please respond with a short greeting.")
            logger.info(f"响应: {response[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ 创建 {model} 代理失败: {e}")
            logger.error(traceback.format_exc())

def test_create_openai_agent_for_opponent():
    """测试通过create_openai_agent_for_opponent函数创建代理"""
    logger.info("=== 测试通过create_openai_agent_for_opponent创建代理 ===")
    
    for name, opponent_name in OPPONENT_NAMES.items():
        logger.info(f"尝试为对手 {opponent_name} 创建代理")
        try:
            # 使用patch_collector_for_openai中的函数创建代理
            agent = create_openai_agent_for_opponent(opponent_name, openai_config)
            logger.info(f"✅ 成功为 {opponent_name} 创建代理")
            
            # 测试简单请求
            logger.info(f"测试 {opponent_name} 的简单请求...")
            response = agent.__call__("Hello, please respond with a short greeting.")
            logger.info(f"响应: {response[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ 为 {opponent_name} 创建代理失败: {e}")
            logger.error(traceback.format_exc())

def test_ray_model_registry():
    """测试Ray模型注册表和固定对手添加"""
    logger.info("=== 测试Ray模型注册表和固定对手添加 ===")
    
    try:
        # 初始化Ray
        logger.info("初始化Ray...")
        ray.init(ignore_reinit_error=True)
        
        # 创建跟踪器
        logger.info("创建跟踪器...")
        tracker = unstable.Tracker.options(name="TestTracker").remote(
            run_name=f"AgentTest-{int(time.time())}", 
            wandb_project="UnstableBaselines"
        )
        
        # 初始化模型注册表
        logger.info("初始化模型注册表...")
        model_registry = unstable.ModelRegistry.options(name="TestModelRegistry").remote(tracker=tracker)
        ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
        
        # 添加所有OpenAI代理作为固定对手
        for name, opponent_name in OPPONENT_NAMES.items():
            logger.info(f"添加固定对手: {opponent_name}")
            try:
                ray.get(model_registry.add_fixed.remote(name=opponent_name))
                logger.info(f"✅ 成功添加固定对手: {opponent_name}")
            except Exception as e:
                logger.error(f"❌ 添加固定对手 {opponent_name} 失败: {e}")
                logger.error(traceback.format_exc())
        
        # 获取所有模型并验证
        logger.info("获取所有模型并验证...")
        all_models = ray.get(model_registry.get_all_models.remote())
        
        for uid, model_meta in all_models.items():
            logger.info(f"模型 UID: {uid}, 类型: {model_meta.kind}, 名称: {model_meta.path_or_name}")
            
        # 测试模型采样器
        logger.info("测试模型采样器...")
        model_sampler = unstable.samplers.model_samplers.FixedOpponentModelSampler(model_registry=model_registry)
        
        # 采样对手
        for _ in range(5):
            try:
                uid, kind, lora_path, openrouter_name = model_sampler.sample_opponent()
                logger.info(f"采样对手: UID={uid}, 类型={kind}, LoRA路径={lora_path}, 名称={openrouter_name}")
            except Exception as e:
                logger.error(f"❌ 采样对手失败: {e}")
                logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"❌ Ray测试失败: {e}")
        logger.error(traceback.format_exc())
    finally:
        # 关闭Ray
        ray.shutdown()

def test_patched_collector():
    """测试打补丁的收集器"""
    logger.info("=== 测试打补丁的收集器 ===")
    
    try:
        # 初始化Ray
        logger.info("初始化Ray...")
        ray.init(ignore_reinit_error=True)
        
        # 应用补丁
        logger.info("应用OpenAI代理补丁...")
        patch_collector_for_openai(openai_config)
        logger.info("✅ 成功应用补丁")
        
        # 创建一个简单的游戏规格
        from unstable._types import AgentSpec, GameSpec
        
        # 为每个对手创建一个游戏规格
        for name, opponent_name in OPPONENT_NAMES.items():
            logger.info(f"为对手 {opponent_name} 创建游戏规格")
            
            try:
                # 创建代理规格
                agent_specs = [
                    AgentSpec(
                        pid=0,
                        kind="checkpoint",
                        collect_data=True,
                        lora_path=None,
                        prompt_template="qwen3-no-reasoning",
                        action_extraction_fn="default"
                    ),
                    AgentSpec(
                        pid=1,
                        kind="openrouter",  # 使用openrouter作为kind
                        collect_data=False,
                        openrouter_name=opponent_name,  # 使用带有openai-前缀的名称
                        prompt_template="qwen3-no-reasoning",
                        action_extraction_fn="default"
                    )
                ]
                
                # 创建游戏规格
                game_spec = GameSpec(
                    game_idx=0,
                    env_id="ThreePlayerIPD-v0",
                    seed=42,
                    agent_specs=agent_specs
                )
                
                logger.info(f"游戏规格创建成功: {game_spec}")
                
                # 模拟patched_run_game_impl的行为
                logger.info("模拟patched_run_game_impl的行为...")
                
                # 检查代理规格
                for agent_spec in game_spec.agent_specs:
                    if agent_spec.openrouter_name and agent_spec.openrouter_name.startswith("openai-"):
                        logger.info(f"发现OpenAI代理: {agent_spec.openrouter_name}")
                        
                        try:
                            # 尝试创建代理
                            agent = create_openai_agent_for_opponent(agent_spec.openrouter_name, openai_config)
                            logger.info(f"✅ 成功为 {agent_spec.openrouter_name} 创建代理")
                            
                            # 测试简单请求
                            logger.info(f"测试 {agent_spec.openrouter_name} 的简单请求...")
                            response = agent.__call__("Hello, please respond with a short greeting.")
                            logger.info(f"响应: {response[:100]}...")
                            
                        except Exception as e:
                            logger.error(f"❌ 为 {agent_spec.openrouter_name} 创建代理失败: {e}")
                            logger.error(traceback.format_exc())
                
            except Exception as e:
                logger.error(f"❌ 为 {opponent_name} 创建游戏规格失败: {e}")
                logger.error(traceback.format_exc())
        
    except Exception as e:
        logger.error(f"❌ 补丁测试失败: {e}")
        logger.error(traceback.format_exc())
    finally:
        # 关闭Ray
        ray.shutdown()

def check_environment_variables():
    """检查环境变量"""
    logger.info("=== 检查环境变量 ===")
    
    # 检查OpenAI API密钥
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        logger.info("✅ OPENAI_API_KEY 已设置")
        # 显示密钥的前几个和后几个字符
        masked_key = f"{api_key[:5]}...{api_key[-4:]}"
        logger.info(f"密钥: {masked_key}")
    else:
        logger.error("❌ OPENAI_API_KEY 未设置")
    
    # 检查OpenAI基础URL
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        logger.info(f"✅ OPENAI_BASE_URL 已设置: {base_url}")
    else:
        logger.info("ℹ️ OPENAI_BASE_URL 未设置，将使用默认URL")

if __name__ == "__main__":
    logger.info("开始测试OpenAI代理创建和固定对手注册")
    
    # 检查环境变量
    check_environment_variables()
    
    # 测试直接创建代理
    test_direct_agent_creation()
    
    # 测试通过create_openai_agent_for_opponent创建代理
    test_create_openai_agent_for_opponent()
    
    # 测试Ray模型注册表
    test_ray_model_registry()
    
    # 测试打补丁的收集器
    test_patched_collector()
    
    logger.info("测试完成")
