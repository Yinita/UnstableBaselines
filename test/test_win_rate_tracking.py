"""
测试胜率统计系统的功能
"""
import pytest
import os
import ray
from unittest.mock import Mock, patch
from unstable.trackers import Tracker
from unstable._types import PlayerTrajectory, GameInformation


class TestWinRateTracking:
    """测试胜率统计功能"""
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_ray(self):
        """设置 Ray 环境"""
        if not ray.is_initialized():
            ray.init(local_mode=True)
        yield
        if ray.is_initialized():
            ray.shutdown()
    
    @pytest.fixture
    def tracker(self):
        """创建测试用的 Tracker 实例"""
        with patch.dict(os.environ, {"RECORD_MODELS": "gpt-4o,claude-3"}):
            tracker = Tracker.remote(
                run_name="test_win_rate",
                wandb_project="test",
                use_wandb=False
            )
            return tracker
    
    @pytest.fixture
    def sample_trajectory(self):
        """创建测试用的玩家轨迹"""
        return PlayerTrajectory(
            pid=0,
            final_reward=1.0,  # 获胜
            obs=["test observation"],
            actions=["test action"],
            extracted_actions=["test extracted"],
            format_feedbacks=[{"valid": True}],
            step_infos=[{}],
            game_info={},
            num_turns=1,
            logps=[0.5]
        )
    
    @pytest.fixture
    def sample_game_info(self):
        """创建测试用的游戏信息"""
        return GameInformation(
            game_idx=1,
            pid=[0, 1],
            names={0: "learner", 1: "gpt-4o"},
            final_rewards={0: 1.0, 1: -1.0},
            eval_model_pid=0,
            eval_opponent_name="gpt-4o"
        )
    
    def test_record_models_parsing(self, tracker):
        """测试 RECORD_MODELS 环境变量解析"""
        record_models = ray.get(tracker.record_models.remote()) if hasattr(tracker, 'record_models') else ray.get(tracker.__getattribute__.remote('record_models'))
        assert "gpt-4o" in record_models
        assert "claude-3" in record_models
        assert len(record_models) == 2
    
    def test_track_win_rate_training(self, tracker):
        """测试训练阶段胜率统计"""
        # 模拟获胜
        opponent_info = {"name": "gpt-4o"}
        ray.get(tracker._track_win_rate.remote("train", True, opponent_info))
        
        # 检查统计数据
        data = ray.get(tracker._data.remote()) if hasattr(tracker, '_data') else ray.get(tracker.__getattribute__.remote('_data'))
        assert "core/train/win_rate_overall" in data
        assert "core/train/win_rate_vs_gpt-4o" in data
        assert data["core/train/win_rate_overall"][-1] == 1
        assert data["core/train/win_rate_vs_gpt-4o"][-1] == 1
    
    def test_get_win_rate_stats(self, tracker):
        """测试胜率统计获取"""
        # 添加一些测试数据
        ray.get(tracker._track_win_rate.remote("train", True, {"name": "gpt-4o"}))
        ray.get(tracker._track_win_rate.remote("train", False, {"name": "gpt-4o"}))
        ray.get(tracker._track_win_rate.remote("train", True, {"name": "claude-3"}))
        
        stats = ray.get(tracker.get_win_rate_stats.remote("train"))
        
        assert "overall" in stats
        assert "vs_gpt-4o" in stats
        assert "vs_claude-3" in stats
        assert abs(stats["overall"] - 2/3) < 0.01  # 2胜1负
        assert abs(stats["vs_gpt-4o"] - 0.5) < 0.01  # 1胜1负
        assert abs(stats["vs_claude-3"] - 1.0) < 0.01  # 1胜0负
    
    def test_add_player_trajectory_with_opponent_info(self, tracker, sample_trajectory):
        """测试添加玩家轨迹时的胜率统计"""
        opponent_info = {"name": "gpt-4o"}
        
        ray.get(tracker.add_player_trajectory.remote(sample_trajectory, "test_env", opponent_info))
        
        # 检查胜率统计是否被调用
        data = ray.get(tracker.__getattribute__.remote('_data'))
        assert "core/train/win_rate_overall" in data
        assert "core/train/win_rate_vs_gpt-4o" in data
    
    def test_add_eval_game_information_with_opponent(self, tracker, sample_game_info):
        """测试添加评估游戏信息时的胜率统计"""
        ray.get(tracker.add_eval_game_information.remote(sample_game_info, "test_env"))
        
        # 检查胜率统计是否被调用
        data = ray.get(tracker.__getattribute__.remote('_data'))
        assert "core/eval/win_rate_overall" in data
        assert "core/eval/win_rate_vs_gpt-4o" in data


class TestWinRateIntegration:
    """测试胜率统计的集成功能"""
    
    def test_environment_variable_integration(self):
        """测试环境变量集成"""
        test_models = "model1,model2,model3"
        with patch.dict(os.environ, {"RECORD_MODELS": test_models}):
            tracker = Tracker(
                run_name="test_integration",
                wandb_project="test",
                use_wandb=False
            )
            
            expected_models = {"model1", "model2", "model3"}
            assert tracker.record_models == expected_models
    
    def test_empty_record_models(self):
        """测试空的 RECORD_MODELS 环境变量"""
        with patch.dict(os.environ, {"RECORD_MODELS": ""}):
            tracker = Tracker(
                run_name="test_empty",
                wandb_project="test",
                use_wandb=False
            )
            
            assert tracker.record_models == set()
    
    def test_missing_record_models_env(self):
        """测试缺失的 RECORD_MODELS 环境变量"""
        with patch.dict(os.environ, {}, clear=True):
            tracker = Tracker(
                run_name="test_missing",
                wandb_project="test",
                use_wandb=False
            )
            
            assert tracker.record_models == set()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
