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
    pid: int; obs: str; act: str; reward: float; env_id: str # track env_id for normalization at the end