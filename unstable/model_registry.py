import ray, copy, trueskill, os
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Dict, List

from unstable._types import ModelMeta
from unstable.utils import setup_logger


@ray.remote
class ModelRegistry:
    def __init__(self, tracker, beta: float = 4.0):
        self.TS = trueskill.TrueSkill(beta=beta)
        self._db: dict[str, ModelMeta] = {}
        self._match_counts = defaultdict(int) # (uid_a, uid_b) -> n
        self._exploration = defaultdict(lambda: defaultdict(dict))
        self._current_ckpt_uid : str | None = None 
        self._tracker = tracker; self._update_step: int = 1
        self.logger = setup_logger("model_registry", ray.get(self._tracker.get_log_dir.remote()))

    @staticmethod
    def _scores_to_ranks(scores: List[float]) -> List[int]:
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        ranks = [0]*len(scores); rank = 0
        for i, idx in enumerate(order):
            if i and scores[idx] != scores[order[i-1]]: rank = i  # next rank starts here
            ranks[idx] = rank
        return ranks

    def add_checkpoint(self, uid: str, path: str, iteration: int, inherit: bool=True):
        self.logger.info(f"tryin to add ckpt: {uid}, path {path}, iteration {iteration}, inherit: {inherit}")
        if uid in self._db: return
        
        # 确保路径是绝对路径
        if path and not os.path.isabs(path):
            # 如果是相对路径，转换为绝对路径
            abs_path = os.path.abspath(path)
            self.logger.info(f"Converting relative path {path} to absolute path {abs_path}")
            path = abs_path
        
        rating = self.TS.Rating(mu=self._db[self._current_ckpt_uid].rating.mu, sigma=self._db[self._current_ckpt_uid].rating.sigma*2) if (inherit and self._current_ckpt_uid in self._db) else self.TS.create_rating()
        self._db[uid] = ModelMeta(uid=uid, kind="checkpoint", path_or_name=path, rating=rating, iteration=iteration)
        self._current_ckpt_uid = uid # make it current
        self.logger.info(f"added ckpt: {uid}, path {path}, iteration {iteration}, inherit: {inherit}")

    def get_all_models(self): return copy.deepcopy(self._db)
    def get_current_ckpt(self) -> str|None: return self._current_ckpt_uid
    def get_name_or_lora_path(self, uid: str) -> str: return self._db[uid].path_or_name
    def add_fixed(self, name: str, prior_mu: float = 25.): 
        if f"fixed-{name}" not in self._db: self._db[f"fixed-{name}"] = ModelMeta(f"fixed-{name}", "fixed", name, self.TS.create_rating(mu=prior_mu))

    def update_ratings(self, uids: List[str], scores: List[float], env_id: str, dummy_uid: str="fixed-env") -> None:
        # 1) 基本校验：长度一致
        if not uids or not scores or len(uids) != len(scores):
            self.logger.warning(f"[TS] Invalid inputs: uids={uids}, scores={scores}")
            return

        # 2) 过滤 NaN/Inf 分数
        clean_pairs = [(u, s) for u, s in zip(uids, scores) if isinstance(s, (int, float)) and (s == s) and (abs(s) != float('inf'))]
        if not clean_pairs:
            self.logger.warning(f"[TS] All scores invalid. Skip. env={env_id}")
            return

        # 3) 合并重复 UID（同一 UID 在一局中出现多次会导致 TrueSkill 异常）。取平均分聚合。
        agg_scores = defaultdict(list)
        for u, s in clean_pairs:
            agg_scores[u].append(float(s))
        unique_uids = list(agg_scores.keys())
        unique_scores = [sum(agg_scores[u])/len(agg_scores[u]) for u in unique_uids]

        # 4) 至少需要两名唯一选手，否则补一个 dummy 对手
        if len(unique_uids) == 1:
            if dummy_uid not in self._db:
                self.add_fixed(name=dummy_uid.replace("fixed-", ""), prior_mu=25.0)
            unique_uids = [unique_uids[0], dummy_uid]
            unique_scores = [unique_scores[0], 0.0]

        # 5) 确保所有选手存在于数据库
        missing = [u for u in unique_uids if u not in self._db]
        if missing:
            self.logger.warning(f"[TS] Missing uids in registry: {missing}. Skip this update. env={env_id}")
            return

        rating_groups = [[self._db[uid].rating] for uid in unique_uids]
        ranks = self._scores_to_ranks(unique_scores)
        try:
            new_groups = self.TS.rate(rating_groups, ranks=ranks)
        except Exception as e:
            # 安全回退：记录上下文并跳过该次更新，避免中断调度
            self.logger.error(f"[TS] rate() failed: {e}. env={env_id}, uids={unique_uids}, scores={unique_scores}, ranks={ranks}")
            return

        # flatten, then write back
        for uid, (new_rating,) in zip(unique_uids, new_groups):
            self._db[uid].rating = new_rating
            self._db[uid].games += 1
            if ranks[unique_uids.index(uid)] == 0:
                self._db[uid].wins  += 1
            elif ranks.count(ranks[unique_uids.index(uid)]) > 1:
                self._db[uid].draws += 1

        # update pair-wise match matrix for analysis/debugging
        for i, uid_i in enumerate(unique_uids):
            for uid_j in unique_uids[i+1:]:
                self._match_counts[tuple(sorted((uid_i, uid_j)))] += 1
        self._update_step += 1

        # push to tracker every n update steps
        if not self._update_step%10: self._tracker.log_model_registry.remote(ts_dict={uid: asdict(meta) for uid, meta in self._db.items()}, match_counts=copy.deepcopy(self._match_counts))

