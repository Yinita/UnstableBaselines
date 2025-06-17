
import trueskill, math, ray, random, time, logging
from collections import defaultdict
from typing import List

# local imports
from unstable.core import Opponent
from unstable.utils.logging import setup_error_logger


@ray.remote
class ModelPool:
    def __init__(self, sample_mode, max_active_lora, tracker=None, lag_range=(1,7), beta=4.0):
        self.TS = trueskill.TrueSkill(beta=beta)
        self._models   = {} # uid -> Opponent dataclass
        self._ckpt_log = [] # ordered list of checkpoint uids
        self.lag_lo, self.lag_hi = lag_range
        self.sample_mode = sample_mode # "self-play", "lagged", "fixed", "adaptive-trueskill"
        self.max_active_lora = max_active_lora
        self._latest_uid = None
        self._last_ckpt = None

        # for tracking
        self._match_counts = defaultdict(int) # (uid_a, uid_b) -> games played
        self._unique_actions_seqs = {}
        self._full_unique_actions_seqs = {
            "unigrams": set(), "bigrams": set(), "trigrams": set(), "4-grams": set(), "5-grams": set(),
            "total_counts": {"unigrams": 0, "bigrams": 0, "trigrams": 0, "4-grams": 0, "5-grams": 0,
        }}
        self._step_counter = 0 # learner step snapshot id
        self._tracker = tracker

        # set up logging
        log_dir = ray.get(tracker.get_log_dir.remote())
        self.logger = setup_error_logger("model_pool", log_dir)

    def current_uid(self):      return self._latest_uid
    def latest_ckpt(self):      return self._ckpt_log[-1] if self._ckpt_log else None
    def ckpt_path(self, uid):   return None if uid is None else  self._models[uid].path_or_name

    def add_checkpoint(self, path: str, iteration: int):
        uid = f"ckpt-{iteration}"
        # inherit μ/σ if a previous checkpoint exists
        if self._latest_uid and self._latest_uid in self._models:
            init_rating = self.TS.Rating(mu=self._models[self._latest_uid].rating.mu, sigma=self._models[self._latest_uid].rating.sigma * 2)
        else:
            init_rating = self.TS.create_rating()   # default prior

        self._models[uid] = Opponent(uid, "checkpoint", path, rating=init_rating)
        self._ckpt_log.append(uid)
        self._latest_uid = uid # promote to “current”
        self._maintain_active_pool() # update current ckpt pool

    def add_fixed(self, name, prior_mu=25.0):
        uid = f"fixed-{name}"
        if uid not in self._models:
            self._models[uid] = Opponent(uid, "fixed", name, rating=self.TS.create_rating(prior_mu))

    def sample(self, uid_me):
        match self.sample_mode:
            case "fixed":           return self._sample_fixed_opponent()                        # randomly sample one of the fixed opponents provided
            case "mirror":          return self.latest_ckpt()                                   # literally play against yourself
            case "lagged":          return self._sample_lagged_opponent()                       # sample an opponent randomly from the available checkpoints (will be lagged by default)
            case "random":          return self._sample_random_opponent()                       # randomly select an opponent from prev checkpoints and fixed opponents
            case "match-quality":   return self._sample_match_quality_opponent(uid_me=uid_me)   # sample an opponent (fixed or prev) based on the TrueSkill match quality
            case "ts-dist":         return self._sample_ts_dist_opponent(uid_me=uid_me)         # sample an opponent (fixed or prev) based on the absolute difference in TrueSkill scores
            case "exploration":     return self._sample_exploration_opponent(uid_me=uid_me)     # sample an opponent based on the expected number of unique board states when playing against that opponent
            case _:                 raise ValueError(self.sample_mode)

    def _sample_fixed_opponent(self):   return random.choice([u for u,m in self._models.items() if m.kind=="fixed"])
    def _sample_lagged_opponent(self):  return random.choice([u for u,m in self._models.items() if (m.kind=="checkpoint" and m.active==True)])
    def _sample_random_opponent(self):  return random.choice([u for u,m in self._models.items() if m.kind=="fixed"]+[u for u,m in self._models.items() if (m.kind=="checkpoint" and m.active==True)])

    def _sample_match_quality_opponent(self, uid_me: str) -> str:
        """ Pick an opponent with probability ∝ TrueSkill match-quality """
        cand, weights = [], []
        for uid, opp in self._models.items():
            if uid == uid_me or not opp.active:
                continue
            q = self.TS.quality([self._models[uid_me].rating, opp.rating]) # ∈ (0,1]
            cand.append(uid)
            weights.append(q) # already scaled

        # softmax the weights
        for i in range(len(weights)): weights[i] = weights[i] / sum(weights)
        if not cand: return None
        return random.choices(cand, weights=weights, k=1)[0]

    def _sample_ts_dist_opponent(self, uid_me: str) -> str:
        """Sample by *absolute* TrueSkill μ-distance (closer ⇒ higher prob)."""
        cand, weights = [], []
        for uid, opp in self._models.items():
            if uid == uid_me or not opp.active:
                continue
            d = abs(self._models[uid_me].rating.mu - opp.rating.mu)
            cand.append(uid)
            weights.append(d)

        for i in range(len(weights)):
            weights[i] = 1 - (weights[i] / sum(weights)) # smaller dist greater match prob
        if not cand:
            return None
        return random.choices(cand, weights=weights, k=1)[0]

    def _sample_exploration_opponent(self, uid_me: str): raise NotImplementedError

    def _update_ratings(self, uid_me: str, uid_opp: str, final_reward: float):
        a = self._models[uid_me].rating
        b = self._models[uid_opp].rating

        if final_reward == 1:       new_a, new_b = self.TS.rate_1vs1(a, b) # uid_me wins → order is (a, b)
        elif final_reward == -1:    new_b, new_a = self.TS.rate_1vs1(b, a) # uid_opp wins → order is (b, a)
        elif final_reward == 0:     new_a, new_b = self.TS.rate_1vs1(a, b, drawn=True) # draw
        else: return # unexpected reward value

        self._models[uid_me].rating = new_a
        self._models[uid_opp].rating = new_b

    def _register_game(self, uid_me, uid_opp):
        if uid_opp is None: uid_opp = uid_me # self-play
        pair = tuple(sorted((uid_me, uid_opp)))
        self._match_counts[tuple(sorted((uid_me, uid_opp)))] += 1

    def _track_exploration(self, uid_me: str, uid_opp: str, game_action_seq: List[str]):
        key = tuple(sorted((uid_me, uid_opp)))
        if key not in self._unique_actions_seqs:
            self._unique_actions_seqs[key] = {
                "unigrams": set(), "bigrams": set(), "trigrams": set(), "4-grams": set(), "5-grams": set(),
                "total_counts": {"unigrams": 0, "bigrams": 0, "trigrams": 0, "4-grams": 0, "5-grams": 0}
            }

        for n, name in [(1,"unigrams"), (2, "bigrams"), (3, "trigrams"), (4, "4-grams"), (5, "5-grams")]:
            for i in range(len(game_action_seq) - n + 1):
                self._unique_actions_seqs[key][name].add(tuple(game_action_seq[i:i+n]))
                self._unique_actions_seqs[key]["total_counts"][name] += 1
                self._full_unique_actions_seqs[name].add(tuple(game_action_seq[i:i+n]))
                self._full_unique_actions_seqs["total_counts"][name] += 1


    def push_game_outcome(self, uid_me: str, uid_opp: str, final_reward: float, game_action_seq: List[str]):
        if uid_me not in self._models or uid_opp not in self._models: return  # skip if either side is unknown
        self._update_ratings(uid_me=uid_me, uid_opp=uid_opp, final_reward=final_reward) # update ts
        self._register_game(uid_me=uid_me, uid_opp=uid_opp) # register the game for tracking
        # self._track_exploration(uid_me=uid_me, uid_opp=uid_opp, game_action_seq=game_action_seq) # tracke unique action seqs
        self._track_exploration(uid_me, uid_opp, game_action_seq)
        self.snapshot(self._step_counter)
        
    def _exp_win(self, A, B):   return self.TS.cdf((A.mu - B.mu) / ((2*self.TS.beta**2 + A.sigma**2 + B.sigma**2) ** 0.5))
    def _activate(self, uid):   self._models[uid].active = True
    def _retire(self, uid):     self._models[uid].active = False

    def _maintain_active_pool(self):
        current = self.latest_ckpt()
        if current is None: return # nothing to do yet

        # collect candidate ckpts (exclude current, fixed models)
        cands = [uid for uid, m in self._models.items() if m.kind == "checkpoint" and uid != current]
        if not cands: return # only the current ckpt exists

        cur_rating = self._models[current].rating

        # score candidates according to sampling mode
        scores = {}
        match self.sample_mode:
            case "random" | "lagged":   scores = {uid: self._ckpt_log.index(uid) for uid in cands}
            case "match-quality":       scores.update({uid: self.TS.quality([cur_rating, self._models[uid].rating]) for uid in cands})
            case "ts-dist":             scores.update({uid: -abs(cur_rating.mu - self._models[uid].rating.mu) for uid in cands})
            case _:                     scores = {uid: self._ckpt_log.index(uid) for uid in cands}

        # pick the N best, plus the current ckpt
        keep = {current} | set(sorted(scores, key=scores.__getitem__, reverse=True)[:max(0, self.max_active_lora - 1)])

        # flip active flags
        for uid, opp in self._models.items():
            if opp.kind != "checkpoint": continue
            opp.active = (uid in keep)
    
    def _get_exploration_ratios(self):
        stats = {}
        for key in ["unigrams", "bigrams", "trigrams", "4-grams", "5-grams"]: stats[f"unique-counts-{key}"] = len(self._full_unique_actions_seqs[key])
        return stats

    # def snapshot(self, iteration: int):
    #     self._step_counter = iteration
    #     self._tracker.log_model_pool.remote(
    #         step=iteration, match_counts=dict(self._match_counts), 
    #         ts_dict={uid: {"mu": opp.rating.mu, "sigma": opp.rating.sigma} for uid,opp in self._models.items()},
    #         exploration=self._get_exploration_ratios(),
    #     )
    def snapshot(self, iteration: int):
        try:
            self._tracker.log_model_pool.remote(
                step=iteration, match_counts=dict(self._match_counts),
                ts_dict={u: {"mu": o.rating.mu, "sigma": o.rating.sigma} for u, o in self._models.items()},
                exploration=self._get_exploration_ratios(),
            )
        except Exception as exc:
            self.logger.exception(f"failed pushing snapshot at iter={iteration}-\n\n{exc}\n\n")


    def get_snapshot(self):
        latest = self.latest_ckpt()
        r = self._models[latest].rating if latest else self.TS.create_rating()
        return {"num_ckpts": len(self._ckpt_log), "mu": r.mu, "sigma": r.sigma}
