from dataclasses import dataclass, field
from typing import List, Dict
import trueskill, math, ray, random

@dataclass
class Trajectory:
    pid: List[int] = field(default_factory=list)
    obs: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    extracted_actions: List[str] = field(default_factory=list)
    infos: List[Dict] = field(default_factory=list)
    final_rewards: Dict[int, float] = field(default_factory=dict)
    num_turns: int = field(default_factory=int)
    format_feedbacks: List[Dict] = field(default_factory=list)
    board_states: List[str] = field(default_factory=list)


@dataclass
class Step:
    pid: int; obs: str; act: str; reward: float; env_id: str; raw_reward: float; transformed_end_of_game_reward: float; step_reward: float

@dataclass
class Opponent:
    uid: str # “ckpt-1234” or “gemini-flash”
    kind: str # {"checkpoint","fixed"}
    path_or_name: str # LoRA dir or OpenRouter model id
    rating: trueskill.Rating # trueskill.Rating(mu, sigma)
    active: bool = True

@ray.remote
class ModelPool:
    def __init__(self, sample_mode, max_active_lora, lag_range=(1,7), beta=4.0):
        self.TS = trueskill.TrueSkill(beta=beta)
        self._models   = {}         # uid -> Opponent dataclass
        self._ckpt_log = []         # ordered list of checkpoint uids
        self.lag_lo, self.lag_hi = lag_range
        self.sample_mode = sample_mode #"self-play", "lagged", "fixed", "adaptive-trueskill"
        self.max_active_lora = max_active_lora
        self._latest_uid = None

    def current_uid(self):
        return self._latest_uid

    def add_checkpoint(self, path: str, iteration: int):
        uid = f"ckpt-{iteration}"

        # inherit μ/σ if a previous checkpoint exists
        if self._latest_uid and self._latest_uid in self._models:
            prev_rating = self._models[self._latest_uid].rating
            init_rating = self.TS.Rating(mu=prev_rating.mu, sigma=25/3) #prev_rating.sigma)
        else:
            init_rating = self.TS.create_rating()   # default prior

        self._models[uid] = Opponent(uid, "checkpoint", path, rating=init_rating)
        self._ckpt_log.append(uid)
        self._latest_uid = uid                     # promote to “current”

    def add_fixed(self, name, prior_mu=25.0):
        uid = f"fixed-{name}"
        if uid not in self._models:
            self._models[uid] = Opponent(uid, "fixed", name, rating=self.TS.create_rating(prior_mu))

    def latest_ckpt(self):
        return self._ckpt_log[-1] if self._ckpt_log else None

    def ckpt_path(self, uid):
        if uid is None: return None
        return self._models[uid].path_or_name

    def sample(self, uid_me, band=(0.2,0.4,0.6,0.8)):
        if self.sample_mode == "self-play":
            return None #self.latest_ckpt()
        if self.sample_mode == "lagged":
            if len(self._ckpt_log) <= self.lag_lo:
                return self._ckpt_log[0]
            lo = max(0, len(self._ckpt_log)-self.lag_hi)
            hi = len(self._ckpt_log)-self.lag_lo
            return random.choice(self._ckpt_log[lo:hi])
        if self.sample_mode == "fixed":
            fixed = [u for u,m in self._models.items() if m.kind=="fixed"]
            return random.choice(fixed)
        if self.sample_mode == "adaptive-trueskill":
            return self._sample_trueskill(uid_me, band)
        raise ValueError(self.sample_mode)


    def update_ratings(self, uid_me, uid_opp, final_reward):
        if uid_me not in self._models or uid_opp not in self._models:
            return  # skip if either side is unknown

        a = self._models[uid_me].rating
        b = self._models[uid_opp].rating

        if final_reward == 1:
            # uid_me wins → order is (a, b)
            new_a, new_b = self.TS.rate_1vs1(a, b)
        elif final_reward == -1:
            # uid_opp wins → order is (b, a)
            new_b, new_a = self.TS.rate_1vs1(b, a)
        elif final_reward == 0:
            # draw
            new_a, new_b = self.TS.rate_1vs1(a, b, drawn=True)
        else:
            return  # unexpected reward value

        self._models[uid_me].rating = new_a
        self._models[uid_opp].rating = new_b
        print(f"\nUPDATED TRUESKILL:\n\t{uid_me} -> {self._models[uid_me].rating}\n\t{uid_opp} -> {self._models[uid_opp].rating}")

    def _exp_win(self, A, B):
        return self.TS.cdf((A.mu - B.mu) / ((2*self.TS.beta**2 + A.sigma**2 + B.sigma**2) ** 0.5))

    def _sample_trueskill(self, uid_me, win_exp_bounds):
        (l1,h1,l2,h2) = win_exp_bounds
        me = self._models[uid_me].rating
        cand = []
        for uid, m in self._models.items():
            if not m.active: continue
            if uid == uid_me: continue
            p = self._exp_win(me, m.rating)
            if l1<=p<=h1 or l2<=p<=h2:
                cand.append((uid, abs(0.5-p)))
        if not cand:                # fallback closest-to-50 %
            cand = [(uid, abs(0.5-self._exp_win(me,m.rating))) for uid,m in self._models.items() if uid!=uid_me]
        weights = [math.exp(-10*d) for _,d in cand]
        return random.choices([u for u,_ in cand], weights)[0]


    def _activate(self, uid): self._models[uid].active = True
    def _retire(self, uid): self._models[uid].active = False
    def _maintain_active_pool(self):
        active_uids = [u for u,m in self._models.items() if m.active and m.kind=="checkpoint"]
        if len(active_uids) <= MAX_ACTIVE: return
        # choose who to retire: oldest OR weakest below threshold
        to_drop = sorted(active_uids, key=lambda u: (self._models[u].rating.mu, self._ckpt_log.index(u)))[:len(active_uids)-MAX_ACTIVE]           # weakest first, then oldest
        for u in to_drop: self._retire(u)