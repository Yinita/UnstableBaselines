from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Trajectory:
    pid: List[int] = field(default_factory=list)
    obs: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    final_rewards: Dict[int, float] = field(default_factory=dict)
    num_turns: int = field(default_factory=int)
    format_feedbacks: List[Dict] = field(default_factory=list)


@dataclass
class Step:
    pid: int; obs: str; act: str; reward: float; env_id: str 



@ray.remote
class PathRegistry:
    def __init__(self): self._paths = []
    def add(self, ckpt_path): self._paths.append(ckpt_path)
    def latest(self): return self._paths[-1]
    def sample(self, lo=1, hi=5):
        window = self._paths[max(0, len(self._paths)-hi):-lo or None]
        return random.choice(window) if window else self.latest()
do s