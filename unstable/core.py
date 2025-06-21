import os, ray, torch, hashlib, datetime, trueskill
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter, deque

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

@dataclass
class Step:
    pid: int
    obs: str 
    act: str
    reward: float
    env_id: str
    step_info: Dict

@dataclass
class Opponent:
    uid: str # “ckpt-1234” or “gemini-flash”
    kind: str # {"checkpoint","fixed"}
    path_or_name: str # LoRA dir or OpenRouter model id
    rating: trueskill.Rating # trueskill.Rating(mu, sigma)
    active: bool = True

@dataclass
class EpisodeResult:
    traj: Trajectory
    end_by_opponent_invalid: bool
    action_seq: List[str]
    final_rewards: List[float]

@dataclass(frozen=True)
class PlaySpec:
    env_id: str
    num_players: int
    player_id: int
    agent_specs: List
    seed: int

@dataclass(frozen=True)
class AgentSpec:
    kind: str # "checkpoint" | "openrouter"
    model: str | None = None  # LoRA dir or OpenRouter model name
    prompt_template: str = "default" # prompt template key
    action_extraction_fn: str = "default"

@dataclass
class TaskMeta:
    type: str  # "train" | "eval"
    env_id: str
    player_id: int
    seed: int
    ckpt_uid: str | None = None
    opponent_uid: str | None = None

class BaseAlgo:
    def initialize(self, model, tokenizer, device, max_train_len: Optional[int]= None, accelerator=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_train_len = max_train_len
        self.accel = accelerator

    def prepare_batch(self, steps):
        """ Turn a list[Step] into tensors on self.dev. Return whatever update() needs """
        raise NotImplementedError

    def update(self, batch):
        """ One gradient update on *this worker only*. Must call .backward() but NOT .step(). Return latest loss as float (for logging) """
        raise NotImplementedError

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
    
    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str): raise NotImplementedError
    def add_eval_episode(self, episode_info: Dict, final_reward: int, player_id: int, env_id: str, iteration: int): raise NotImplementedError
    def log_lerner(self, info_dict: Dict): raise NotImplementedError

class ExplorationTracker:
    def __init__(self, window: int = 512, ngram_sizes: Tuple[int, ...] = (1, 2, 3, 4)):
        self.window = window
        self.ngram_sizes = ngram_sizes
        self._iter = 0
        self.counter: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter)) # env -> n -> Counter
        self.total: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int)) # env -> n -> int
        self.games: Dict[str, Dict[int, deque]] = defaultdict(lambda: defaultdict(deque)) # env -> n -> deque

    def _ngrams(self, toks: List[str], n: int) -> List[int]:    return [hash(tuple(toks[i : i + n])) for i in range(len(toks) - n + 1)]
    def pct_unique(self, env_id: str, n: int) -> float:         return len(self.counter[env_id][n]) / self.total[env_id][n] if self.total[env_id][n] else 0.0
    def add_game(self, action_seq: List[str], env_id: str) -> None:
        self._iter += 1
        for n in self.ngram_sizes:
            ng_hashes = self._ngrams(action_seq, n)
            bucket_c  = self.counter[env_id][n]
            for h in ng_hashes: bucket_c[h] += 1
            self.total[env_id][n] += len(ng_hashes)
            self.games[env_id][n].append((self._iter, ng_hashes))
            cutoff = self._iter - self.window
            dq = self.games[env_id][n]
            while dq and dq[0][0] <= cutoff:
                _, old_hashes = dq.popleft()
                for h in old_hashes:
                    bucket_c[h] -= 1
                    if bucket_c[h] == 0: del bucket_c[h]
                self.total[env_id][n] -= len(old_hashes)