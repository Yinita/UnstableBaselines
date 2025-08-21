import os, re, ray, time, wandb, collections, datetime, logging, numpy as np
from collections import defaultdict
from typing import Dict, Optional, Union, List
from unstable.utils.logging import setup_logger
from unstable.utils.misc import write_game_information_to_file
from unstable._types import PlayerTrajectory, GameInformation
import wandb
Scalar = Union[int, float, bool]

class BaseTracker:
    def __init__(self, run_name: str):
        self.run_name = run_name 
        self._build_output_dir()

    def _build_output_dir(self):
        self.output_dir = os.path.join("outputs", str(datetime.datetime.now().strftime('%Y-%m-%d')), str(datetime.datetime.now().strftime('%H-%M-%S')), self.run_name)
        os.makedirs(self.output_dir)
        self.output_dirs = {}
        for folder_name in ["training_data", "eval_data", "checkpoints", "logs"]: 
            self.output_dirs[folder_name] =  os.path.join(self.output_dir, folder_name); os.makedirs(self.output_dirs[folder_name], exist_ok=True)

    def get_checkpoints_dir(self):  return self.output_dirs["checkpoints"]
    def get_train_dir(self):        return self.output_dirs["training_data"]
    def get_eval_dir(self):         return self.output_dirs["eval_data"]
    def get_log_dir(self):          return self.output_dirs["logs"]
    def add_trajectory(self, trajectory: PlayerTrajectory, env_id: str): raise NotImplementedError
    def add_eval_episode(self, episode_info: Dict, final_reward: int, player_id: int, env_id: str, iteration: int): raise NotImplementedError
    def log_lerner(self, info_dict: Dict): raise NotImplementedError

    
@ray.remote
class Tracker(BaseTracker): 
    FLUSH_EVERY = 64
    def __init__(self, run_name: str, wandb_project: Optional[str]=None):
        super().__init__(run_name=run_name)
        self.logger = setup_logger("tracker", self.get_log_dir())
        self.use_wandb = False
        if wandb_project: wandb.init(project=wandb_project, name=run_name); self.use_wandb = True; wandb.define_metric("*", step_metric="learner/step")
        self._m: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=512))
        self._buffer: Dict[str, Scalar] = {}
        self._n = {}
        self._last_flush = time.monotonic()
        self._interface_stats = {"gpu_tok_s": {}, "TS": {}, "exploration": {}, "match_counts": {}, "format_success": None, "inv_move_rate": None, "game_len": None}
        
        # 胜率统计相关
        self._record_models = self._parse_record_models()
        self._win_rate_stats = {
            "train": {"all": [], "vs_specific": {model: [] for model in self._record_models}},
            "eval": {"all": [], "vs_specific": {model: [] for model in self._record_models}}
        }
        self.logger.info(f"胜率统计初始化，记录模型: {self._record_models}")
    
    def _parse_record_models(self) -> List[str]:
        """解析环境变量中的记录模型列表"""
        record_models_env = os.environ.get("RECORD_MODELS", "")
        if not record_models_env:
            return []
        return [model.strip() for model in record_models_env.split(",") if model.strip()]

    def _put(self, k: str, v: Scalar): self._m[k].append(v)
    def _is_scalar(self, x) -> bool:
        return isinstance(x, (int, float, bool, np.number))

    def _mean_of_deque(self, dq: collections.deque) -> Optional[float]:
        vals = [float(v) for v in dq if self._is_scalar(v)]
        return float(np.mean(vals)) if vals else None

    def _agg(self, p: str) -> dict[str, Scalar]:
        out: Dict[str, Scalar] = {}
        for k, dq in self._m.items():
            if k.startswith(p):
                m = self._mean_of_deque(dq)
                if m is not None:
                    out[k] = m
        return out
    def _flush_if_due(self):
        if time.monotonic()-self._last_flush >= self.FLUSH_EVERY:
            if self._buffer and self.use_wandb:
                try: wandb.log(self._buffer)
                except Exception as e: self.logger.warning(f"wandb.log failed: {e}")
            self._buffer.clear(); self._last_flush=time.monotonic()

    def add_player_trajectory(self, traj: PlayerTrajectory, env_id: str, opponent_info: Optional[Dict] = None):
        try:
            reward = traj.final_reward; player_id = traj.pid
            self._put(f"collection-{env_id}/reward", reward)
            self._put(f"collection-{env_id}/Win Rate (pid={traj.pid})", int(reward>0))
            self._put(f"collection-{env_id}/Draw (pid={traj.pid})", int(reward==0))
            self._put(f"collection-{env_id}/Reward (pid={traj.pid})", reward)
            self._put(f"collection-{env_id}/Game Length", traj.num_turns)
            for idx in range(len(traj.obs)):
                self._put(f"collection-{env_id}/Respone Length (char)", len(traj.actions[idx]))
                self._put(f"collection-{env_id}/Observation Length (char)", len(traj.obs[idx]))
                for k, v in traj.format_feedbacks[idx].items(): self._put(f"collection-{env_id}/Format Success Rate - {k}", v)
            self._n[f"collection-{env_id}"] = self._n.get(f"collection-{env_id}", 0) + 1
            self._put(f"collection-{env_id}/step", self._n[f"collection-{env_id}"])

            # Global (cross-env) unified metrics under 'collection/' prefix
            self._put("collection/reward", reward)
            self._put("collection/Win Rate", int(reward>0))
            self._put("collection/Draw", int(reward==0))
            self._put(f"collection/Reward (pid={traj.pid})", reward)
            self._put("collection/Game Length", traj.num_turns)
            for idx in range(len(traj.obs)):
                self._put("collection/Respone Length (char)", len(traj.actions[idx]))
                self._put("collection/Observation Length (char)", len(traj.obs[idx]))
                for k, v in traj.format_feedbacks[idx].items(): self._put(f"collection/Format Success Rate - {k}", v)
            self._n["collection"] = self._n.get("collection", 0) + 1
            self._put("collection/step", self._n["collection"]) 

            # 胜率统计 - 训练阶段
            self._track_win_rate("train", reward > 0, opponent_info)

            # Aggregate both per-env and global prefixes
            self._buffer.update(self._agg('collection-'))
            self._buffer.update(self._agg('collection/'))
            self._flush_if_due()
        except Exception as exc:
            self.logger.info(f"Exception when adding trajectory to tracker: {exc}")

    def add_eval_game_information(self, game_information: GameInformation, env_id: str):
        try:
            eval_reward = game_information.final_rewards.get(game_information.eval_model_pid, 0.0)
            _prefix = f"evaluation-{env_id}" if not game_information.eval_opponent_name else f"evaluation-{env_id} ({game_information.eval_opponent_name})"
            # Only log metrics for the evaluated model pid
            self._put(f"{_prefix}/Reward (pid={game_information.eval_model_pid})", eval_reward)
            self._put(f"{_prefix}/Win Rate (pid={game_information.eval_model_pid})",  int(eval_reward>0))
            self._put(f"{_prefix}/Draw Rate (pid={game_information.eval_model_pid})", int(eval_reward==0))
            self._n[_prefix] = self._n.get(_prefix, 0) + 1
            self._put(f"{_prefix}/step", self._n[_prefix])

            # Optionally, keep global metrics only for the evaluated pid
            self._put(f"evaluation/Reward (pid={game_information.eval_model_pid})", eval_reward)
            self._put(f"evaluation/Win Rate (pid={game_information.eval_model_pid})",  int(eval_reward>0))
            self._put(f"evaluation/Draw Rate (pid={game_information.eval_model_pid})", int(eval_reward==0))
            self._n["evaluation"] = self._n.get("evaluation", 0) + 1
            self._put("evaluation/step", self._n["evaluation"]) 

            # 胜率统计 - 评估阶段
            opponent_info = {"name": game_information.eval_opponent_name} if game_information.eval_opponent_name else None
            self._track_win_rate("eval", eval_reward > 0, opponent_info)

            # Aggregate both per-env and global prefixes
            self._buffer.update(self._agg('evaluation-'))
            self._buffer.update(self._agg('evaluation/'))
            self._flush_if_due()

            # try storing the eval info to file
            write_game_information_to_file(game_info=game_information, filename=os.path.join(self.get_eval_dir(), f"{env_id}-{game_information.game_idx}.csv"))

        except Exception as exc:
            self.logger.info(f"Exception when adding game_info to tracker: {exc}")

    def _track_win_rate(self, phase: str, is_win: bool, opponent_info: Optional[Dict] = None):
        """追踪胜率统计
        
        Args:
            phase: "train" 或 "eval"
            is_win: 是否获胜
            opponent_info: 对手信息，包含 name 字段
        """
        try:
            # 总体胜率统计
            self._put(f"core/{phase}/win_rate_overall", int(is_win))
            
            # 对特定对手的胜率统计
            if opponent_info and "name" in opponent_info:
                opponent_name = opponent_info["name"]
                if opponent_name in self._record_models:
                    self._put(f"core/{phase}/win_rate_vs_{opponent_name}", int(is_win))
            
            # 更新统计计数
            self._n[f"core/{phase}"] = self._n.get(f"core/{phase}", 0) + 1
            self._put(f"core/{phase}/step", self._n[f"core/{phase}"])
            
        except Exception as exc:
            self.logger.warning(f"Exception in win rate tracking: {exc}")

    def get_win_rate_stats(self, phase: str) -> Dict[str, float]:
        """获取胜率统计信息
        
        Args:
            phase: "train" 或 "eval"
            
        Returns:
            胜率统计字典
        """
        stats = {}
        try:
            # 总体胜率
            overall_key = f"core/{phase}/win_rate_overall"
            if overall_key in self._m:
                wins = sum(1 for x in self._m[overall_key] if x)
                total = len(self._m[overall_key])
                stats["overall"] = wins / total if total > 0 else 0.0
            
            # 对特定对手的胜率
            for model_name in self._record_models:
                model_key = f"core/{phase}/win_rate_vs_{model_name}"
                if model_key in self._m:
                    wins = sum(1 for x in self._m[model_key] if x)
                    total = len(self._m[model_key])
                    stats[f"vs_{model_name}"] = wins / total if total > 0 else 0.0
                    
        except Exception as exc:
            self.logger.warning(f"Exception getting win rate stats: {exc}")
            
        return stats

    def log_win_rate_summary(self, phase: str):
        """记录胜率摘要到日志"""
        try:
            stats = self.get_win_rate_stats(phase)
            if stats:
                self.logger.info(f"Win rate summary ({phase}):")
                for key, rate in stats.items():
                    self.logger.info(f"  {key}: {rate:.3f}")
        except Exception as exc:
            self.logger.warning(f"Exception logging win rate summary: {exc}")

    def log_model_registry(self, ts_dict: dict[str, dict[str, float]], match_counts: dict[tuple[str, str], int]):
        self._interface_stats.update({"TS": ts_dict, "exploration": None, "match_counts": match_counts})

    def log_inference(self, actor: str, gpu_ids: list[int], stats: dict[str, float]):
        for key in stats: self._put(f"inference/{actor}/{key}", stats[key])
        for gpu_id in gpu_ids: self._interface_stats["gpu_tok_s"][gpu_id] = stats["tok_s"]
        self._buffer.update(self._agg('inference'))
    
    def log_learner(self, info: dict):
        try:
            for k, v in info.items():
                base_key = f"learner/{k}"
                if self._is_scalar(v):
                    self._put(base_key, v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if self._is_scalar(vv):
                            self._put(f"{base_key}/{kk}", vv)
                # Non-scalar and non-dict values are ignored
            self._buffer.update(self._agg("learner")); self._flush_if_due()
        except Exception as exc:
            self.logger.info(f"Exception in log_learner: {exc}")

    def get_interface_info(self): 
        for inf_key in ["Game Length", "Format Success Rate - correct_answer_format", "Format Success Rate - invalid_move"]: 
            vals = []
            for k, dq in self._m.items():
                if inf_key in k:
                    m = self._mean_of_deque(dq)
                    if m is not None:
                        vals.append(m)
            self._interface_stats[inf_key] = float(np.mean(vals)) if vals else None
        return self._interface_stats
