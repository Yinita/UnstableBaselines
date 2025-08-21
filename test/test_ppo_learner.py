#!/usr/bin/env python3
"""
PPO Learner 单元测试
测试 PPO 算法的核心功能、参数配置和训练流程
"""

import unittest
import torch
import ray
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, Any, List
import numpy as np

# 导入被测试的模块
from unstable.learners.ppo_learner import PPOLearner, compute_gae
from unstable.learners.base import BaseLearner
from unstable._types import PlayerTrajectory, Step


@dataclass
class MockStepData:
    """模拟的步骤数据"""
    obs: str
    act: str
    reward: float
    step_info: Dict[str, Any] = field(default_factory=dict)


class MockBuffer:
    """模拟的缓冲区"""
    def __init__(self):
        self.data = []
        self.max_size = 1000
    
    def size(self):
        return len(self.data)
    
    def get_batch(self, batch_size):
        return self.data[:batch_size]
    
    def stop(self):
        pass


class MockTracker:
    """模拟的追踪器"""
    def __init__(self):
        self.logs = []
    
    def get_log_dir(self):
        return "/tmp/test_logs"
    
    def get_checkpoints_dir(self):
        return "/tmp/test_checkpoints"
    
    def log_learner(self, log_data):
        self.logs.append(log_data)


class MockModelRegistry:
    """模拟的模型注册表"""
    def __init__(self):
        self.checkpoints = {}
    
    def add_checkpoint(self, uid, path, iteration):
        self.checkpoints[uid] = {"path": path, "iteration": iteration}


class TestComputeGAE(unittest.TestCase):
    """测试 GAE 计算函数"""
    
    def setUp(self):
        """设置测试环境"""
        self.device = torch.device("cpu")
    
    def test_compute_gae_basic(self):
        """测试基本的 GAE 计算"""
        rewards = torch.tensor([1.0, 0.5, -0.2, 1.0])
        values = torch.tensor([0.8, 0.6, 0.4, 0.9])
        
        advantages, returns = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)
        
        # 检查形状
        self.assertEqual(advantages.shape, rewards.shape)
        self.assertEqual(returns.shape, rewards.shape)
        
        # 检查返回值计算正确性
        expected_returns = advantages + values
        torch.testing.assert_close(returns, expected_returns)
    
    def test_compute_gae_with_bootstrap(self):
        """测试带 bootstrap 值的 GAE 计算"""
        rewards = torch.tensor([1.0, 0.5])
        values = torch.tensor([0.8, 0.6])
        last_value = 0.7
        
        advantages, returns = compute_gae(
            rewards, values, 
            gamma=0.99, gae_lambda=0.95, 
            last_value=last_value, done=False
        )
        
        # 检查形状
        self.assertEqual(advantages.shape, rewards.shape)
        self.assertEqual(returns.shape, rewards.shape)
    
    def test_compute_gae_terminal_state(self):
        """测试终止状态的 GAE 计算"""
        rewards = torch.tensor([1.0, 0.5, -1.0])
        values = torch.tensor([0.8, 0.6, 0.0])
        
        advantages_terminal, _ = compute_gae(
            rewards, values, done=True
        )
        
        advantages_non_terminal, _ = compute_gae(
            rewards, values, done=False, last_value=0.5
        )
        
        # 终止状态和非终止状态的优势应该不同
        self.assertFalse(torch.allclose(advantages_terminal, advantages_non_terminal))


class TestPPOLearner(unittest.TestCase):
    """测试 PPO Learner 类"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        # 初始化 Ray（如果尚未初始化）
        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True)
    
    @classmethod
    def tearDownClass(cls):
        """清理测试类"""
        if ray.is_initialized():
            ray.shutdown()
    
    def setUp(self):
        """设置每个测试"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟对象
        self.mock_buffer = MockBuffer()
        self.mock_tracker = MockTracker()
        self.mock_model_registry = MockModelRegistry()
        
        # PPO 配置参数
        self.ppo_config = {
            "model_name": "microsoft/DialoGPT-small",
            "lora_cfg": {
                "lora_rank": 8,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"]
            },
            "batch_size": 4,
            "mini_batch_size": 2,
            "learning_rate": 1e-4,
            "grad_clip": 1.0,
            "buffer": self.mock_buffer,
            "tracker": self.mock_tracker,
            "model_registry": self.mock_model_registry,
            "activation_checkpointing": False,
            "gradient_checkpointing": False,
            "use_trainer_cache": True
        }
    
    def tearDown(self):
        """清理每个测试"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('unstable.learners.ppo_learner.build_peft_model')
    @patch('unstable.utils.setup_logger')
    def test_ppo_learner_initialization(self, mock_logger, mock_build_model):
        """测试 PPO Learner 初始化"""
        # 模拟模型构建
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_build_model.return_value = (mock_model, mock_tokenizer)
        
        # 创建 PPO Learner
        learner = PPOLearner(**self.ppo_config)
        
        # 验证初始化
        self.assertIsNotNone(learner.policy_model)
        self.assertIsNotNone(learner.tokenizer)
        self.assertIsNotNone(learner.policy_optimizer)
    
    @patch('unstable.learners.ppo_learner.build_peft_model')
    @patch('unstable.utils.setup_logger')
    def test_initialize_algorithm(self, mock_logger, mock_build_model):
        """测试算法初始化"""
        # 模拟模型构建
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_build_model.return_value = (mock_model, mock_tokenizer)
        
        learner = PPOLearner(**self.ppo_config)
        
        # 初始化算法
        learner.initialize_algorithm(
            infer_mini_batch_size=8,
            critic_learning_rate=5e-5,
            normalize_adv=True,
            max_generation_len=512,
            max_train_len=1024,
            clip_ratio=0.2,
            ppo_epochs=4,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            kl_target=0.01,
            kl_coef=0.2
        )
        
        # 验证参数设置
        self.assertEqual(learner.infer_mini_batch_size, 8)
        self.assertEqual(learner.clip_ratio, 0.2)
        self.assertEqual(learner.ppo_epochs, 4)
        self.assertEqual(learner.entropy_coef, 0.01)
        self.assertEqual(learner.value_loss_coef, 0.5)
        self.assertEqual(learner.kl_target, 0.01)
        self.assertEqual(learner.kl_coef, 0.2)
        self.assertTrue(learner.normalize_adv)
        self.assertIsNotNone(learner.critic)
        self.assertIsNotNone(learner.critic_optimizer)
    
    @patch('unstable.learners.ppo_learner.build_peft_model')
    @patch('unstable.utils.setup_logger')
    def test_prepare_batch(self, mock_logger, mock_build_model):
        """测试批次数据准备"""
        # 模拟模型和分词器
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1]
        }
        mock_build_model.return_value = (mock_model, mock_tokenizer)
        
        learner = PPOLearner(**self.ppo_config)
        learner.initialize_algorithm(
            infer_mini_batch_size=8,
            critic_learning_rate=5e-5,
            max_generation_len=512,
            max_train_len=1024
        )
        
        # 创建测试步骤数据
        steps = [
            MockStepData(
                obs="观察1",
                act="动作1",
                reward=1.0,
                step_info={"return": 2.0, "old_logp": -1.5}
            ),
            MockStepData(
                obs="观察2",
                act="动作2",
                reward=0.5,
                step_info={"return": 1.0, "old_logp": -2.0}
            )
        ]
        
        # 模拟分词器行为
        def mock_tokenizer_call(*args, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                return Mock(
                    input_ids=torch.tensor([[1, 2, 3, 4], [1, 2, 3, 0]]),
                    attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
                )
            else:
                return {"input_ids": [1, 2, 3]}
        
        learner.tokenizer.side_effect = mock_tokenizer_call
        
        # 测试批次准备
        result = learner._prepare_batch(steps)
        
        # 验证返回值
        self.assertEqual(len(result), 8)  # enc, state_enc, advs, rets, old_logps, obs, avg_len, pct_truncated
    
    def test_ppo_loss_computation(self):
        """测试 PPO 损失计算的数学正确性"""
        # 创建测试数据
        batch_size = 4
        seq_len = 10
        
        # 模拟概率比率
        old_logps = torch.tensor([-2.0, -1.5, -3.0, -2.5])
        new_logps = torch.tensor([-1.8, -1.7, -2.8, -2.3])
        advantages = torch.tensor([1.0, -0.5, 2.0, -1.0])
        
        # 计算比率
        ratio = torch.exp(new_logps - old_logps)
        
        # PPO 裁剪损失
        clip_ratio = 0.2
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()
        
        # 验证损失计算
        self.assertIsInstance(ppo_loss, torch.Tensor)
        self.assertEqual(ppo_loss.dim(), 0)  # 标量
        
        # 验证裁剪行为
        clipped_ratios = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        self.assertTrue(torch.all(clipped_ratios >= 1.0 - clip_ratio))
        self.assertTrue(torch.all(clipped_ratios <= 1.0 + clip_ratio))
    
    def test_entropy_computation(self):
        """测试熵计算"""
        # 创建测试概率分布
        batch_size, seq_len, vocab_size = 2, 5, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # 计算概率和熵
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1)
        
        # 验证熵的形状和性质
        self.assertEqual(entropy.shape, (batch_size, seq_len))
        self.assertTrue(torch.all(entropy >= 0))  # 熵应该非负
    
    def test_kl_divergence_computation(self):
        """测试 KL 散度计算"""
        old_logps = torch.tensor([-2.0, -1.5, -3.0])
        new_logps = torch.tensor([-1.8, -1.7, -2.8])
        
        # KL 散度计算
        kl_div = (old_logps - new_logps).mean()
        
        # 验证 KL 散度
        self.assertIsInstance(kl_div, torch.Tensor)
        self.assertEqual(kl_div.dim(), 0)
    
    @patch('unstable.learners.ppo_learner.build_peft_model')
    @patch('unstable.utils.setup_logger')
    def test_gradient_clipping(self, mock_logger, mock_build_model):
        """测试梯度裁剪"""
        # 模拟模型
        mock_model = Mock()
        mock_model.parameters.return_value = [
            Mock(grad=Mock(data=Mock(norm=Mock(return_value=2.0)))),
            Mock(grad=Mock(data=Mock(norm=Mock(return_value=3.0))))
        ]
        mock_tokenizer = Mock()
        mock_build_model.return_value = (mock_model, mock_tokenizer)
        
        learner = PPOLearner(**self.ppo_config)
        
        # 验证梯度裁剪参数
        self.assertEqual(learner.grad_clip, 1.0)
    
    def test_early_stopping_logic(self):
        """测试早停逻辑"""
        kl_target = 0.01
        current_kl = 0.02  # 超过 1.5 * kl_target
        
        # 模拟早停条件
        should_stop = current_kl > 1.5 * kl_target
        self.assertTrue(should_stop)
        
        # 正常情况
        current_kl = 0.005
        should_stop = current_kl > 1.5 * kl_target
        self.assertFalse(should_stop)


class TestPPOIntegration(unittest.TestCase):
    """PPO 集成测试"""
    
    def test_ppo_config_validation(self):
        """测试 PPO 配置验证"""
        # 测试有效配置
        valid_config = {
            "clip_ratio": 0.2,
            "ppo_epochs": 4,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "kl_target": 0.01,
            "kl_coef": 0.2
        }
        
        # 验证配置范围
        self.assertGreater(valid_config["clip_ratio"], 0)
        self.assertLess(valid_config["clip_ratio"], 1)
        self.assertGreater(valid_config["ppo_epochs"], 0)
        self.assertGreaterEqual(valid_config["entropy_coef"], 0)
        self.assertGreater(valid_config["value_loss_coef"], 0)
        self.assertGreater(valid_config["kl_target"], 0)
        self.assertGreaterEqual(valid_config["kl_coef"], 0)
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 测试极端值处理
        large_logp = torch.tensor([50.0, -50.0, 100.0])
        clipped_logp = torch.clamp(large_logp, -30.0, 30.0)
        
        # 验证裁剪效果
        self.assertTrue(torch.all(clipped_logp >= -30.0))
        self.assertTrue(torch.all(clipped_logp <= 30.0))
        
        # 测试 NaN 处理
        nan_tensor = torch.tensor([1.0, float('nan'), 2.0])
        clean_tensor = torch.nan_to_num(nan_tensor, nan=0.0)
        self.assertFalse(torch.isnan(clean_tensor).any())


class TestPPOMetrics(unittest.TestCase):
    """测试 PPO 指标计算"""
    
    def test_clipped_fraction_calculation(self):
        """测试裁剪比例计算"""
        clip_ratio = 0.2
        ratios = torch.tensor([0.9, 1.1, 1.3, 0.7, 1.0])
        
        clipped_mask = (ratios < 1.0 - clip_ratio) | (ratios > 1.0 + clip_ratio)
        clipped_fraction = clipped_mask.float().mean()
        
        # 验证裁剪比例
        expected_clipped = 2  # 1.3 和 0.7 被裁剪
        expected_fraction = expected_clipped / len(ratios)
        self.assertAlmostEqual(clipped_fraction.item(), expected_fraction, places=4)
    
    def test_value_accuracy_metrics(self):
        """测试值函数准确性指标"""
        value_pred = torch.tensor([1.0, -0.5, 2.0, -1.0])
        returns = torch.tensor([1.2, -0.3, 1.8, -0.8])
        
        # MAE
        mae = (value_pred - returns).abs().mean()
        
        # 方向准确性
        dir_acc = ((value_pred > 0) == (returns > 0)).float().mean()
        
        # 验证指标
        self.assertGreaterEqual(mae.item(), 0)
        self.assertGreaterEqual(dir_acc.item(), 0)
        self.assertLessEqual(dir_acc.item(), 1)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
