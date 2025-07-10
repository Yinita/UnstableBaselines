import ray, random
from typing import Dict, Any, Tuple, Optional

from unstable.types import GameInformation


class BaseModelSampler:
    def __init__(self, model_registry): 
        self.model_registry = model_registry
    
    def get_current_ckpt(self):         
        current_ckpt_uid = ray.get(self.model_registry.get_current_ckpt.remote())
        current_ckpt_lora_path = ray.get(self.model_registry.get_name_or_lora_path.remote(uid=current_ckpt_uid))
        return current_ckpt_uid, current_ckpt_lora_path
    
    def update(self, game_info: GameInformation, job_info: Dict[str, Any]):
        self.model_registry.update_ratings.remote(
            uids = [m["uid"] for m in job_info["models"] if m["pid"] in game_info.final_rewards],
            scores = [game_info.final_rewards[m["pid"]] for m in job_info["models"] if m["pid"] in game_info.final_rewards],
            env_id = job_info["env_id"]
        )
    def sample_opponent(self, *, current_uid: str, registry_snapshot: dict) -> Tuple[str, str, Optional[str], Optional[str]]: raise NotImplementedError

class FixedOpponentModelSampler(BaseModelSampler):
    def sample_opponent(self, *, current_uid: str, registry_snapshot: dict) -> Tuple[str, str, Optional[str], Optional[str]]:
        opponent_meta = random.choice([model_meta for uid, model_meta in ray.get(self.model_registry.get_all_models.remote()) if model_meta.active and model_meta.kind=="fixed"])
        return (opponent_meta.uid, opponent_meta.kind, None, opponent_meta.path_or_name) 

