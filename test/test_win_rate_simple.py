"""
简化的胜率统计系统测试
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from unstable._types import PlayerTrajectory, GameInformation


def test_record_models_parsing():
    """测试 RECORD_MODELS 环境变量解析"""
    with patch.dict(os.environ, {"RECORD_MODELS": "gpt-4o,claude-3,gemini"}):
        # 模拟 Tracker 的 _parse_record_models 方法
        record_models_str = os.environ.get("RECORD_MODELS", "")
        if record_models_str:
            record_models = set(model.strip() for model in record_models_str.split(",") if model.strip())
        else:
            record_models = set()
        
        assert "gpt-4o" in record_models
        assert "claude-3" in record_models
        assert "gemini" in record_models
        assert len(record_models) == 3


def test_empty_record_models():
    """测试空的 RECORD_MODELS 环境变量"""
    with patch.dict(os.environ, {"RECORD_MODELS": ""}):
        record_models_str = os.environ.get("RECORD_MODELS", "")
        if record_models_str:
            record_models = set(model.strip() for model in record_models_str.split(",") if model.strip())
        else:
            record_models = set()
        
        assert record_models == set()


def test_missing_record_models_env():
    """测试缺失的 RECORD_MODELS 环境变量"""
    with patch.dict(os.environ, {}, clear=True):
        record_models_str = os.environ.get("RECORD_MODELS", "")
        if record_models_str:
            record_models = set(model.strip() for model in record_models_str.split(",") if model.strip())
        else:
            record_models = set()
        
        assert record_models == set()


def test_win_rate_tracking_logic():
    """测试胜率统计逻辑"""
    # 模拟 Tracker 的胜率统计数据结构
    _data = {}
    record_models = {"gpt-4o", "claude-3"}
    
    def _put(key, value):
        if key not in _data:
            _data[key] = []
        _data[key].append(value)
    
    def _track_win_rate(phase, is_win, opponent_info=None):
        # 总体胜率统计
        _put(f"core/{phase}/win_rate_overall", int(is_win))
        
        # 对特定对手的胜率统计
        if opponent_info and "name" in opponent_info:
            opponent_name = opponent_info["name"]
            if opponent_name in record_models:
                _put(f"core/{phase}/win_rate_vs_{opponent_name}", int(is_win))
    
    # 测试训练阶段胜率统计
    _track_win_rate("train", True, {"name": "gpt-4o"})
    _track_win_rate("train", False, {"name": "gpt-4o"})
    _track_win_rate("train", True, {"name": "claude-3"})
    
    # 检查统计数据
    assert "core/train/win_rate_overall" in _data
    assert "core/train/win_rate_vs_gpt-4o" in _data
    assert "core/train/win_rate_vs_claude-3" in _data
    
    # 检查具体数值
    assert _data["core/train/win_rate_overall"] == [1, 0, 1]  # 胜, 负, 胜
    assert _data["core/train/win_rate_vs_gpt-4o"] == [1, 0]  # 胜, 负
    assert _data["core/train/win_rate_vs_claude-3"] == [1]   # 胜


def test_win_rate_stats_calculation():
    """测试胜率统计计算"""
    # 模拟统计数据
    _data = {
        "core/train/win_rate_overall": [1, 0, 1, 1, 0],  # 3胜2负 = 60%
        "core/train/win_rate_vs_gpt-4o": [1, 0, 1],      # 2胜1负 = 66.7%
        "core/train/win_rate_vs_claude-3": [1, 0],       # 1胜1负 = 50%
    }
    record_models = {"gpt-4o", "claude-3"}
    
    def get_win_rate_stats(phase):
        stats = {}
        # 总体胜率
        overall_key = f"core/{phase}/win_rate_overall"
        if overall_key in _data:
            wins = sum(_data[overall_key])
            total = len(_data[overall_key])
            stats["overall"] = wins / total if total > 0 else 0.0
        
        # 对特定对手的胜率
        for model_name in record_models:
            model_key = f"core/{phase}/win_rate_vs_{model_name}"
            if model_key in _data:
                wins = sum(_data[model_key])
                total = len(_data[model_key])
                stats[f"vs_{model_name}"] = wins / total if total > 0 else 0.0
        
        return stats
    
    stats = get_win_rate_stats("train")
    
    assert "overall" in stats
    assert "vs_gpt-4o" in stats
    assert "vs_claude-3" in stats
    assert abs(stats["overall"] - 0.6) < 0.01
    assert abs(stats["vs_gpt-4o"] - 0.667) < 0.01
    assert abs(stats["vs_claude-3"] - 0.5) < 0.01


def test_opponent_info_extraction():
    """测试对手信息提取逻辑"""
    # 模拟 GameInformation
    game_info = GameInformation(
        game_idx=1,
        pid=[0, 1],
        names={0: "learner", 1: "gpt-4o"},
        final_rewards={0: 1.0, 1: -1.0}
    )
    
    # 模拟从 GameInformation 提取对手信息的逻辑
    def extract_opponent_info(game_information, current_pid):
        opponent_info = None
        if game_information.names:
            # 找到对手的名称（非当前玩家的其他玩家）
            opponent_names = [name for pid, name in game_information.names.items() if pid != current_pid]
            if opponent_names:
                opponent_info = {"name": opponent_names[0]}  # 取第一个对手
        return opponent_info
    
    # 测试提取对手信息
    opponent_info = extract_opponent_info(game_info, 0)  # 当前玩家是 pid=0
    assert opponent_info is not None
    assert opponent_info["name"] == "gpt-4o"
    
    opponent_info = extract_opponent_info(game_info, 1)  # 当前玩家是 pid=1
    assert opponent_info is not None
    assert opponent_info["name"] == "learner"


def test_win_rate_aggregation():
    """测试胜率聚合功能"""
    # 模拟原始数据
    _data = {
        "core/train/win_rate_overall": [1, 0, 1, 1, 0],
        "core/train/win_rate_vs_gpt-4o": [1, 0, 1],
        "core/train/win_rate_vs_claude-3": [1, 0],
    }
    
    def _agg(prefix):
        """模拟聚合函数"""
        out = {}
        for k, v in _data.items():
            if k.startswith(prefix):
                if v:  # 只处理非空列表
                    m = sum(v) / len(v)  # 计算平均值
                    out[k] = m
        return out
    
    # 测试聚合
    aggregated = _agg("core/train/")
    
    assert "core/train/win_rate_overall" in aggregated
    assert "core/train/win_rate_vs_gpt-4o" in aggregated
    assert "core/train/win_rate_vs_claude-3" in aggregated
    
    # 检查平均胜率
    assert abs(aggregated["core/train/win_rate_overall"] - 0.6) < 0.01  # 3/5
    assert abs(aggregated["core/train/win_rate_vs_gpt-4o"] - 0.667) < 0.01  # 2/3
    assert abs(aggregated["core/train/win_rate_vs_claude-3"] - 0.5) < 0.01  # 1/2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
