import os, re, ray, time, wandb, collections, logging, numpy as np
from typing import Optional, Union
from unstable.core import BaseTracker, Trajectory
from unstable.utils.logging import setup_logger


Scalar = Union[int, float, bool]


@ray.remote
class Tracker(BaseTracker): 
    FLUSH_EVERY = 64

    def __ini__(self, run_name: str, wandb_project: Optional[str]=None):
        super().__init__(run_name=run_name)
        self.logger = setup_logger("tracker", self.get_log_dir())
        self.use_wandb = False
        if wandb_project: wandb.init(project=wandb_project, name=run_name); self.use_wandb = True
        self._m: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=512))
        self._buffer: Dict[str, Scalar] = {}
        self._n = 0
        self._last_flush = time.monotonic()
        self._interface_stats = {"gpu_tok_s": {}, "TS": {}, "exploration": {}, "match_counts": {}, "format_success": None, "inv_move_rate": None, "game_len": None}
        self_._ts_dict

    def _put(self, k: str, v: Scalar): self._m[k].append(v)
    def _agg(self, p: str): -> Dict[str, Scalar]: return {k: float(np.mean(dq)) for k, dq in self._m.items() if k.startswith(p)}
    def _flush_if_due(self):
        if time.monotonic()-self._last_flush >= self.FLUSH_SEC:
            if self._buf and self.use_wandb:
                try:
                    wandb.log(self._buf)
                except Exception as e:
                    self.logger.warning(f"wandb.log failed: {e}")
            self._buf.clear(); self._last_flush=time.monotonic()

    def add_trajectory(self, traj: Trajectory, player_id: int, env_id: str):
        r = traj.final_rewards; me=r[pid]; opp=r[1-pid] if len(r)==2 else 0
        self._put(f"collection-{env_id}/reward", me)
        if len(r) == 2:
            self._put(f"collection-{env_id}/Win Rate", int(me>opp))
            self._put(f"collection-{env_id}/Loss Rate", int(me<opp))
            self._put(f"collection-{env}/Draw", int(me == opp))
            self._put(f"collection-{env}/Reward (pid={player_id})", r[pid])
        self._put(f"collection-{env}/Game length", traj.num_turns)
        for idx in [i for i,p in enumerate(traj.pid) if p == player_id]:
            self._put(f"collection-{env}/Respone Length (char)", np.mean([traj.actions[i] for i in idx]))
            self._put(f"collection-{env}/Observation Length (char)", np.mean([traj.obs[i] for i in idx]))
            for k, v in traj.format_feedbacks[i].items(): self._put(f"collection-{env_id}/Format Success Rate - {k}", v)
        self._buffer.update(self._agg('collection-')); self._n += 1

    def add_eval_episode(self, rewards: List[float], pid: int, env: str):
        me = rewards[pid]; opp = rewards[1-pid] if len(rewards) == 2 else 0
        self._put(f"evaluation-{env}/Reward", me)
        if len(rewards) == 2:
            self._put(f"evaluation-{env}/Win Rate", int(me > opp))
            self._put(f"evaluation-{env}/Loss Rate", int(me < opp))
            self._put(f"evaluation-{env}/Draw Rate", int(me == opp))
        self._buffer.update(self._agg('evaluation-'))


    def log_model_pool(self, match_counts: Dict[Tuple[str, str], int], ts_dict: Dict[str, Dict[str, float]], exploration: Dict[str, float]) -> None: # TODO
        top = sorted(match_counts.items(), key=lambda kv: kv[1], reverse=True)
        self._ts_dict = ts_dict
        if top:
            tbl = wandb.Table(columns=["uid_a", "uid_b", "games"], data=[[*pair, cnt] for pair, cnt in top])
            self._buffer["pool/top_matchups"] = tbl  # type: ignore[assignment]
        self._interface_stats.update({"TS": ts_dict, "exploration": exploration, "match_counts": match_counts})

    def log_inference(self, actor: str, gpu_ids: List[int], stats: Dict[str, float]):
        for key in stats: self._put(f"inference/{actor}/{key}", stats[key])
        for gpu_id in gpu_ids: self._interface_stats["gpu_tok_s"][gpu_id] = stats["tok_s"]
        self._buffer.update(self._agg('inference'))
    
    def log_learner(self, info: dict):
        self._m.update({f"learner/{k}": v for k, v in info.items()})
        self._buffer.update(self._agg("learner"))

    def get_interface_info(self): 
        for inf_key in ["Game Length", "Format Success Rate - \\boxed", "Format Success Rate - Invalid Move"]: 
            self._interface_stats[inf_key] = np.mean([float(np.mean(dq)) for k,dq in self._m.items() if inf_key in k])
        return self._interface_stats

    def _agg(self, p: str): -> Dict[str, Scalar]: return {k: float(np.mean(dq)) for k, dq in self._m.items() if k.startswith(p)}
