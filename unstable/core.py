import os, ray, torch, datetime, trueskill
from dataclasses import dataclass, field
from typing import List, Dict, Optional


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
    agents: List
    seed: int


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

