
import trueskill, math, ray, random, time
from collections import defaultdict
from typing import List

# local imports
from unstable.core import Opponent

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

    def current_uid(self):
        return self._latest_uid

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

    def latest_ckpt(self):
        return self._ckpt_log[-1] if self._ckpt_log else None

    def ckpt_path(self, uid):
        if uid is None: return None
        return self._models[uid].path_or_name

    def sample(self, uid_me):
        if self.sample_mode == "fixed": return self._sample_fixed_opponent() # randomly sample one of the fixed opponents provided
        elif self.sample_mode == "mirror": return self.latest_ckpt() # literally play against yourself
        elif self.sample_mode == "lagged": return self._sample_lagged_opponent() # sample an opponent randomly from the available checkpoints (will be lagged by default)
        elif self.sample_mode == "random": return self._sample_random_opponent() # randomly select an opponent from prev checkpoints and fixed opponents
        elif self.sample_mode == "match-quality": return self._sample_match_quality_opponent(uid_me=uid_me) # sample an opponent (fixed or prev) based on the TrueSkill match quality
        elif self.sample_mode == "ts-dist": return self._sample_ts_dist_opponent(uid_me=uid_me) # sample an opponent (fixed or prev) based on the absolute difference in TrueSkill scores
        elif self.sample_mode == "ts-dist-biased": return self._sample_ts_dist_biased_opponent(uid_me=uid_me) # sample an opponent (fixed or prev) based on the exp(mu_i-mu_b) distance (i.e. biased towards stronger opponents)
        elif self.sample_mode == "exploration": return self._sample_exploration_opponent(uid_me=uid_me) # sample an opponent based on the expected number of unique board states when playing against that opponent
        else: raise ValueError(self.sample_mode)


    def _sample_fixed_opponent(self):
        fixed = [u for u,m in self._models.items() if m.kind=="fixed"]
        return random.choice(fixed)

    def _sample_lagged_opponent(self):
        available = [u for u,m in self._models.items() if (m.kind=="checkpoint" and m.active==True)]
        return random.choice(available)

    def _sample_random_opponent(self):
        fixed = [u for u,m in self._models.items() if m.kind=="fixed"]
        checkpoints = [u for u,m in self._models.items() if (m.kind=="checkpoint" and m.active==True)]
        return random.choice(fixed+checkpoints)

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
        weights_sum = sum(weights)
        for i in range(len(weights)): 
            weights[i] = weights[i] / weights_sum
        # # fallback – nothing active except myself
        if not cand:
            return None
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

        weights_sum = sum(weights)
        for i in range(len(weights)):
            weights[i] = 1 - (weights[i] / weights_sum) # smaller dist greater match prob
        if not cand:
            return None
        return random.choices(cand, weights=weights, k=1)[0]

    def _sample_ts_dist_biased_opponent(self, uid_me: str):
        cand, weights = [], []
        for uid, opp in self._models.items():
            if uid == uid_me or not opp.active:
                continue
            d = math.exp((opp.rating.mu-self._models[uid_me].rating.mu)/25)
            cand.append(uid)
            weights.append(d)

        weights_sum = sum(weights)
        for i in range(len(weights)):
            weights[i] = weights[i]/weights_sum
        if not cand:
            return None
        return random.choices(cand, weights=weights, k=1)[0]

    def _sample_exploration_opponent(self, uid_me: str):
        raise NotImplementedError

    def _update_ratings(self, uid_me: str, uid_opp: str, final_reward: float):
        a = self._models[uid_me].rating
        b = self._models[uid_opp].rating

        if final_reward == 1: new_a, new_b = self.TS.rate_1vs1(a, b) # uid_me wins → order is (a, b)
        elif final_reward == -1: new_b, new_a = self.TS.rate_1vs1(b, a) # uid_opp wins → order is (b, a)
        elif final_reward == 0: new_a, new_b = self.TS.rate_1vs1(a, b, drawn=True) # draw
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
                "total_counts": {"unigrams": 0, "bigrams": 0, "trigrams": 0, "4-grams": 0, "5-grams": 0,
            }}

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
        self._track_exploration(uid_me=uid_me, uid_opp=uid_opp, game_action_seq=game_action_seq) # tracke unique action seqs


    def _exp_win(self, A, B): return self.TS.cdf((A.mu - B.mu) / ((2*self.TS.beta**2 + A.sigma**2 + B.sigma**2) ** 0.5))
    def _activate(self, uid): self._models[uid].active = True
    def _retire(self, uid): self._models[uid].active = False

    def _maintain_active_pool(self):
        current = self.latest_ckpt()
        if current is None: return # nothing to do yet

        # collect candidate ckpts (exclude current, fixed models)
        cands = [uid for uid, m in self._models.items() if m.kind == "checkpoint" and uid != current]
        if not cands: return # only the current ckpt exists

        cur_rating = self._models[current].rating

        # score candidates according to sampling mode
        scores = {}
        if self.sample_mode in {"random", "lagged"}: 
            scores = {uid: self._ckpt_log.index(uid) for uid in cands}

        elif self.sample_mode == "match-quality":
            for uid in cands:
                scores[uid] = self.TS.quality([cur_rating, self._models[uid].rating])

        elif self.sample_mode == "ts-dist":
            for uid in cands:
                scores[uid] = -abs(cur_rating.mu - self._models[uid].rating.mu) 

        elif self.sample_mode == "ts-dist-biased":
            pos, neg = {}, {}
            for uid in cands:
                delta = self._models[uid].rating.mu - cur_rating.mu
                (pos if delta >= 0 else neg)[uid] = delta
            # keep strongest positives first; if not enough, use closest negatives
            ordered = (sorted(pos, key=pos.__getitem__, reverse=True) + sorted(neg, key=lambda u: abs(neg[u])))
            scores = {uid: len(ordered) - i for i, uid in enumerate(ordered)}# bigger rank ⇒ bigger score

        else:
            scores = {uid: self._ckpt_log.index(uid) for uid in cands}

        # pick the N best, plus the current ckpt
        N = self.max_active_lora - 1 # slot for current already reserved
        keep = {current} | set(sorted(scores, key=scores.__getitem__, reverse=True)[:max(0, N)])

        # flip active flags
        for uid, opp in self._models.items():
            if opp.kind != "checkpoint": continue
            opp.active = (uid in keep)

    
    def _get_exploration_ratios(self):
        stats = {}
        # for key, data in self._unique_actions_seqs.items():
        #     ratios = {}
        #     for ngram_type in ["unigrams", "bigrams", "trigrams", "4-grams", "5-grams"]:
        #         total = data["total_counts"].get(ngram_type, 0)
        #         unique = len(data.get(ngram_type, set()))
        #         ratio = unique / total if total > 0 else 0.0
        #         ratios[ngram_type] = ratio
        #     stats[key] = ratios
        
        for key in ["unigrams", "bigrams", "trigrams", "4-grams", "5-grams"]:
            stats[f"unique-counts-{key}"] = len(self._full_unique_actions_seqs[key])

        print(self._full_unique_actions_seqs)
            
        return stats


    def snapshot(self, iteration: int):
        self._step_counter = iteration
        self._tracker.log_model_pool.remote(
            iteration=iteration, match_counts=dict(self._match_counts),
            ts_dict={uid: {"mu": opp.rating.mu, "sigma": opp.rating.sigma} for uid,opp in self._models.items()},
            exploration_ratios=self._get_exploration_ratios()
        )


