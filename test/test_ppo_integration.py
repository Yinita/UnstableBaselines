#!/usr/bin/env python3
"""
PPO Learner 集成测试
测试 PPO 与其他组件的集成和端到端功能
"""

import pytest
import torch
import ray
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from unstable.learners.ppo_learner import PPOLearner
from unstable import runtime


@pytest.mark.integration
class TestPPOIntegrationWithRuntime:
    """测试 PPO 与 runtime 的集成"""
    
    def test_ppo_in_runtime_build(self):
        """测试在 runtime.build 中使用 PPO"""
        # 验证 PPO 在支持的算法列表中
        assert "ppo" in runtime._ALGOS
        assert runtime._ALGOS["ppo"] == PPOLearner
        
        # 验证 PPO 使用正确的缓冲区类型
        assert "ppo" in runtime._EPISODE_BUFFER_ALGOS
    
    @patch('ray.init')
    @patch('unstable.Tracker')
    @patch('unstable.ModelRegistry')
    @patch('unstable.GameScheduler')
    @patch('unstable.EpisodeBuffer')
    @patch('unstable.Collector')
    def test_ppo_runtime_configuration(self, mock_collector, mock_buffer, 
                                     mock_scheduler, mock_registry, 
                                     mock_tracker, mock_ray_init):
        """测试 PPO 在 runtime 中的配置"""
        # 模拟依赖
        mock_tracker_instance = Mock()
        mock_tracker_instance.get_log_dir.return_value = "/tmp/logs"
        mock_tracker.return_value = mock_tracker_instance
        
        mock_registry_instance = Mock()
        mock_registry.return_value = mock_registry_instance
        
        # 模拟环境规范
        train_envs = [Mock()]
        
        # 这里我们不能直接调用 runtime.build，因为它会尝试初始化真实的模型
        # 但我们可以验证配置逻辑
        algorithm = "ppo"
        assert algorithm in runtime._ALGOS
        assert algorithm in runtime._EPISODE_BUFFER_ALGOS


@pytest.mark.integration
class TestPPOWithMockComponents:
    """使用模拟组件测试 PPO 集成"""
    
    @pytest.fixture
    def mock_components(self):
        """创建模拟的组件"""
        components = {
            'buffer': Mock(),
            'tracker': Mock(),
            'model_registry': Mock()
        }
        
        # 配置 tracker
        components['tracker'].get_log_dir.return_value = "/tmp/test_logs"
        components['tracker'].get_checkpoints_dir.return_value = "/tmp/test_checkpoints"
        
        # 配置 buffer
        components['buffer'].size.return_value = 100
        components['buffer'].get_batch.return_value = []
        
        return components
    
    @patch('unstable.learners.ppo_learner.build_peft_model')
    @patch('unstable.utils.setup_logger')
    def test_ppo_learner_with_mock_components(self, mock_logger, mock_build_model, mock_components):
        """测试 PPO Learner 与模拟组件的集成"""
        # 模拟模型构建
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_build_model.return_value = (mock_model, mock_tokenizer)
        
        # 创建 PPO Learner
        learner = PPOLearner(
            model_name="test-model",
            lora_cfg={"lora_rank": 8},
            batch_size=4,
            mini_batch_size=2,
            learning_rate=1e-4,
            grad_clip=1.0,
            buffer=mock_components['buffer'],
            tracker=mock_components['tracker'],
            model_registry=mock_components['model_registry']
        )
        
        # 初始化算法
        learner.initialize_algorithm(
            infer_mini_batch_size=8,
            critic_learning_rate=5e-5,
            normalize_adv=True,
            max_generation_len=512,
            max_train_len=1024
        )
        
        # 验证组件交互
        assert learner.buffer == mock_components['buffer']
        assert learner.tracker == mock_components['tracker']
        assert learner.model_registry == mock_components['model_registry']


@pytest.mark.integration
class TestPPOTrainingLoop:
    """测试 PPO 训练循环集成"""
    
    @pytest.fixture
    def training_setup(self):
        """设置训练环境"""
        # 创建模拟的训练数据
        from test_ppo_learner import MockStepData
        
        episodes = [
            [
                MockStepData(
                    obs="观察1",
                    act="动作1", 
                    reward=1.0,
                    step_info={"return": 2.0, "advantage": 0.5}
                ),
                MockStepData(
                    obs="观察2",
                    act="动作2",
                    reward=0.5,
                    step_info={"return": 1.0, "advantage": 0.2}
                )
            ]
        ]
        
        return {
            'episodes': episodes,
            'batch_size': 2,
            'mini_batch_size': 1
        }
    
    @patch('unstable.learners.ppo_learner.build_peft_model')
    @patch('unstable.utils.setup_logger')
    def test_ppo_update_cycle(self, mock_logger, mock_build_model, training_setup):
        """测试 PPO 更新周期"""
        # 模拟模型和分词器
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # 配置模型输出
        mock_output = Mock()
        mock_output.logits = torch.randn(2, 10, 1000)  # [batch, seq, vocab]
        mock_model.return_value = mock_output
        
        # 配置分词器
        def tokenizer_side_effect(*args, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                return Mock(
                    input_ids=torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 0]]),
                    attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
                )
            else:
                return {"input_ids": [1, 2, 3, 4]}
        
        mock_tokenizer.side_effect = tokenizer_side_effect
        mock_build_model.return_value = (mock_model, mock_tokenizer)
        
        # 创建模拟组件
        mock_buffer = Mock()
        mock_tracker = Mock()
        mock_registry = Mock()
        
        mock_tracker.get_log_dir.return_value = "/tmp/test"
        mock_tracker.get_checkpoints_dir.return_value = "/tmp/test"
        
        # 创建 PPO Learner
        learner = PPOLearner(
            model_name="test-model",
            lora_cfg={"lora_rank": 8},
            batch_size=training_setup['batch_size'],
            mini_batch_size=training_setup['mini_batch_size'],
            learning_rate=1e-4,
            grad_clip=1.0,
            buffer=mock_buffer,
            tracker=mock_tracker,
            model_registry=mock_registry
        )
        
        # 初始化算法
        learner.initialize_algorithm(
            infer_mini_batch_size=8,
            critic_learning_rate=5e-5,
            max_generation_len=5,
            max_train_len=10
        )
        
        # 模拟 critic 输出
        learner.critic.return_value = torch.tensor([[1.0], [0.5]])
        
        # 测试更新过程（这里只测试不会崩溃）
        try:
            # 这个测试主要验证代码结构正确，不会抛出异常
            batch = training_setup['episodes']
            # 由于涉及复杂的模型前向传播，我们主要测试接口正确性
            assert hasattr(learner, '_update')
            assert callable(learner._update)
        except Exception as e:
            pytest.fail(f"PPO 更新过程出现异常: {e}")


@pytest.mark.integration 
class TestPPOMemoryManagement:
    """测试 PPO 内存管理集成"""
    
    def test_gpu_memory_monitoring_integration(self):
        """测试 GPU 内存监控集成"""
        # 验证 PPO Learner 继承了基类的内存监控功能
        from unstable.learners.base import BaseLearner
        
        assert issubclass(PPOLearner, BaseLearner)
        
        # 验证内存监控方法存在
        methods = ['_monitor_gpu_memory', '_cleanup_memory', 'get_gpu_stats']
        for method in methods:
            assert hasattr(BaseLearner, method)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_memory_cleanup_calls(self, mock_empty_cache, mock_cuda_available):
        """测试内存清理调用"""
        mock_cuda_available.return_value = True
        
        # 创建基础学习器实例来测试内存管理
        mock_buffer = Mock()
        mock_tracker = Mock()
        mock_registry = Mock()
        
        mock_tracker.get_log_dir.return_value = "/tmp/test"
        mock_tracker.get_checkpoints_dir.return_value = "/tmp/test"
        
        with patch('unstable.learners.ppo_learner.build_peft_model') as mock_build:
            mock_build.return_value = (Mock(), Mock())
            
            learner = PPOLearner(
                model_name="test-model",
                lora_cfg={"lora_rank": 8},
                batch_size=4,
                mini_batch_size=2,
                learning_rate=1e-4,
                grad_clip=1.0,
                buffer=mock_buffer,
                tracker=mock_tracker,
                model_registry=mock_registry
            )
            
            # 调用内存清理
            learner._cleanup_memory()
            
            # 验证 CUDA 缓存清理被调用
            mock_empty_cache.assert_called()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
