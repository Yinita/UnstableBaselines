#!/usr/bin/env python3
"""
PPO Learner 测试运行脚本
提供便捷的测试执行和结果报告
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests(test_type="all", verbose=False, coverage=False):
    """
    运行 PPO Learner 测试
    
    Args:
        test_type: 测试类型 ("all", "unit", "integration", "fast")
        verbose: 是否显示详细输出
        coverage: 是否生成覆盖率报告
    """
    
    # 基础命令
    cmd = ["python", "-m", "pytest"]
    
    # 添加测试文件
    test_file = "test/test_ppo_learner.py"
    cmd.append(test_file)
    
    # 根据测试类型添加标记
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    
    # 添加详细输出
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # 添加覆盖率
    if coverage:
        cmd.extend([
            "--cov=unstable.learners.ppo_learner",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # 添加其他有用的选项
    cmd.extend([
        "--tb=short",  # 简短的回溯信息
        "--strict-markers",  # 严格标记模式
        "-x"  # 遇到第一个失败就停止
    ])
    
    print(f"运行命令: {' '.join(cmd)}")
    print("-" * 50)
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    try:
        # 运行测试
        result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return False
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return False

def run_specific_test(test_name, verbose=False):
    """运行特定的测试"""
    cmd = [
        "python", "-m", "pytest",
        f"test/test_ppo_learner.py::{test_name}",
        "-v" if verbose else "-q",
        "--tb=short"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    try:
        result = subprocess.run(cmd, cwd=project_root, env=env)
        return result.returncode == 0
    except Exception as e:
        print(f"运行测试 {test_name} 时出错: {e}")
        return False

def list_available_tests():
    """列出可用的测试"""
    cmd = [
        "python", "-m", "pytest",
        "test/test_ppo_learner.py",
        "--collect-only", "-q"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    try:
        result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=True, text=True)
        print("可用的测试:")
        print(result.stdout)
    except Exception as e:
        print(f"列出测试时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="运行 PPO Learner 测试")
    parser.add_argument(
        "--type", "-t",
        choices=["all", "unit", "integration", "fast"],
        default="all",
        help="测试类型"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="生成覆盖率报告"
    )
    parser.add_argument(
        "--test", "-s",
        help="运行特定的测试"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用的测试"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_tests()
        return
    
    if args.test:
        success = run_specific_test(args.test, args.verbose)
    else:
        success = run_tests(args.type, args.verbose, args.coverage)
    
    if success:
        print("\n✅ 所有测试通过!")
        sys.exit(0)
    else:
        print("\n❌ 测试失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()
