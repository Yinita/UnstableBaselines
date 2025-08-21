"""
pytest 配置文件
为 PPO Learner 测试提供共享的 fixtures 和配置
"""

import pytest
import torch
import ray
import tempfile
import shutil
from unittest.mock import Mock
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture(scope="session", autouse=True)
def ray_setup():
    """设置和清理 Ray 环境"""
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_tokenizer():
    """创建模拟的分词器"""
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": [1, 2, 3, 4, 5],
        "attention_mask": [1, 1, 1, 1, 1]
    }
    
    def tokenizer_call(*args, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            return Mock(
                input_ids=torch.tensor([[1, 2, 3, 4], [1, 2, 3, 0]]),
                attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
            )
        else:
            return {"input_ids": [1, 2, 3]}
    
    tokenizer.side_effect = tokenizer_call
    return tokenizer


@pytest.fixture
def mock_model():
    """创建模拟的模型"""
    model = Mock()
    model.parameters.return_value = [
        Mock(grad=Mock(data=Mock(norm=Mock(return_value=2.0)))),
        Mock(grad=Mock(data=Mock(norm=Mock(return_value=3.0))))
    ]
    return model


@pytest.fixture
def ppo_config():
    """PPO 测试配置"""
    return {
        "clip_ratio": 0.2,
        "ppo_epochs": 4,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "kl_target": 0.01,
        "kl_coef": 0.2,
        "normalize_adv": True,
        "infer_mini_batch_size": 8,
        "critic_learning_rate": 5e-5,
        "max_generation_len": 512,
        "max_train_len": 1024
    }


@pytest.fixture
def sample_trajectories():
    """创建示例轨迹数据"""
    from test_ppo_learner import MockStepData
    
    return [
        [
            MockStepData(
                obs="Player 1 观察",
                act="Player 1 动作",
                reward=1.0,
                step_info={"return": 2.0, "advantage": 0.5, "old_logp": -1.5}
            ),
            MockStepData(
                obs="Player 1 观察2",
                act="Player 1 动作2",
                reward=0.5,
                step_info={"return": 1.0, "advantage": 0.2, "old_logp": -2.0}
            )
        ],
        [
            MockStepData(
                obs="Player 2 观察",
                act="Player 2 动作",
                reward=-0.5,
                step_info={"return": 0.5, "advantage": -0.3, "old_logp": -1.8}
            )
        ]
    ]


# 测试标记
def pytest_configure(config):
    """配置 pytest 标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
