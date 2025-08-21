# PPO Learner 测试文档

本目录包含 PPO Learner 的完整测试套件，确保算法实现的正确性和稳定性。

## 测试文件结构

```
test/
├── test_ppo_learner.py          # PPO Learner 核心单元测试
├── test_ppo_integration.py      # PPO 集成测试
├── conftest.py                  # pytest 配置和共享 fixtures
├── run_ppo_tests.py            # 测试运行脚本
└── README_PPO_Tests.md         # 本文档
```

## 测试覆盖范围

### 1. 核心算法测试 (`test_ppo_learner.py`)

#### GAE 计算测试
- ✅ 基本 GAE 计算正确性
- ✅ Bootstrap 值处理
- ✅ 终止状态处理
- ✅ 数值稳定性

#### PPO 算法核心功能
- ✅ PPO Learner 初始化
- ✅ 算法参数配置
- ✅ 批次数据准备
- ✅ PPO 损失计算（裁剪、熵、KL散度）
- ✅ 梯度裁剪
- ✅ 早停机制

#### 数值计算验证
- ✅ 裁剪比率计算
- ✅ 熵计算正确性
- ✅ KL 散度计算
- ✅ 值函数指标

### 2. 集成测试 (`test_ppo_integration.py`)

#### Runtime 集成
- ✅ PPO 在 runtime.build 中的配置
- ✅ 缓冲区类型匹配
- ✅ 算法注册验证

#### 组件集成
- ✅ 与模拟组件的交互
- ✅ 训练循环集成
- ✅ 内存管理集成

## 运行测试

### 方法 1: 使用测试脚本（推荐）

```bash
# 运行所有测试
python test/run_ppo_tests.py

# 运行特定类型的测试
python test/run_ppo_tests.py --type unit
python test/run_ppo_tests.py --type integration
python test/run_ppo_tests.py --type fast

# 详细输出
python test/run_ppo_tests.py --verbose

# 生成覆盖率报告
python test/run_ppo_tests.py --coverage

# 运行特定测试
python test/run_ppo_tests.py --test TestComputeGAE::test_compute_gae_basic

# 列出所有可用测试
python test/run_ppo_tests.py --list
```

### 方法 2: 直接使用 pytest

```bash
# 运行所有 PPO 测试
pytest test/test_ppo_learner.py test/test_ppo_integration.py -v

# 运行单元测试
pytest test/test_ppo_learner.py -v

# 运行集成测试
pytest test/test_ppo_integration.py -v

# 生成覆盖率报告
pytest test/test_ppo_learner.py --cov=unstable.learners.ppo_learner --cov-report=html
```

### 方法 3: 直接运行 Python 文件

```bash
# 运行单元测试
python test/test_ppo_learner.py

# 运行集成测试
python test/test_ppo_integration.py
```

## 测试环境要求

### 必需依赖
- Python 3.8+
- PyTorch
- Ray
- pytest
- unittest.mock

### 可选依赖（用于覆盖率报告）
- pytest-cov

## 测试配置

### 环境变量
- `PYTHONPATH`: 自动设置为项目根目录
- `CUDA_VISIBLE_DEVICES`: 测试中使用 CPU 模式

### Ray 配置
- 使用 `local_mode=True` 进行本地测试
- 自动初始化和清理 Ray 环境

## 测试数据

### 模拟数据结构
- `MockStepData`: 模拟步骤数据
- `MockBuffer`: 模拟数据缓冲区
- `MockTracker`: 模拟日志追踪器
- `MockModelRegistry`: 模拟模型注册表

### 测试参数
```python
PPO_TEST_CONFIG = {
    "clip_ratio": 0.2,
    "ppo_epochs": 4,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "kl_target": 0.01,
    "kl_coef": 0.2,
    "batch_size": 4,
    "mini_batch_size": 2
}
```

## 常见问题

### Q: 测试运行时出现 Ray 初始化错误
A: 确保没有其他 Ray 进程在运行，或使用 `ray.shutdown()` 清理环境。

### Q: 模型加载失败
A: 测试使用模拟模型，不需要真实的预训练模型。确保 mock 配置正确。

### Q: CUDA 相关错误
A: 测试默认使用 CPU 模式，确保 `torch.device("cpu")` 设置正确。

### Q: 内存不足
A: 测试使用小批次大小，如果仍有问题，可以进一步减小测试数据规模。

## 持续集成

建议在 CI/CD 流水线中运行以下测试命令：

```bash
# 快速测试（跳过慢速测试）
python test/run_ppo_tests.py --type fast

# 完整测试套件
python test/run_ppo_tests.py --coverage
```

## 测试指标

### 预期通过率
- 单元测试: 100%
- 集成测试: 100%
- 覆盖率目标: > 90%

### 性能基准
- 单个测试运行时间: < 5秒
- 完整测试套件: < 30秒

## 贡献指南

### 添加新测试
1. 在相应的测试文件中添加测试方法
2. 使用描述性的测试名称
3. 添加适当的测试标记 (`@pytest.mark.unit` 或 `@pytest.mark.integration`)
4. 更新本文档

### 测试最佳实践
- 使用 fixtures 共享测试设置
- 模拟外部依赖
- 测试边界条件和异常情况
- 保持测试独立性
- 添加清晰的断言消息
