import numpy as np
from typing import List, Optional
from collections import defaultdict
from unstable._types import Step

class SamplingRewardTransform:
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]: raise NotImplementedError

class ComposeSamplingRewardTransforms:
    def __init__(self, transforms: List[SamplingRewardTransform]):  self.transforms = transforms
    def __call__(self, steps: List[Step]) -> List[Step]:
        for transform in self.transforms: steps = transform(steps)
        return steps

class NormalizeRewards(SamplingRewardTransform):
    def __init__(self, z_score: bool=False): self.z_score = z_score
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]:
        rewards = np.asarray([step.reward for step in steps], dtype=np.float32)
        rewards = np.nan_to_num(rewards, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        mean, std = np.mean(rewards), np.std(rewards)
        if self.z_score:
            std = np.maximum(std, 1e-8)  # avoid division-by-zero
        for step in steps: step.reward = (step.reward-mean)/(std if self.z_score else 1)
        return steps

class NormalizeRewardsByEnv(SamplingRewardTransform):
    def __init__(self, z_score: bool = False): self.z_score = z_score 
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]:
        env_buckets = defaultdict(list)
        for step in steps: env_buckets[step.env_id].append(step) # bucket by env
        for env_steps in env_buckets.values():
            r = np.asarray([s.reward for s in env_steps], dtype=np.float32)
            r = np.nan_to_num(r, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
            if self.z_score:
                std = np.maximum(np.std(r), 1e-8)  # avoid division-by-zero
                normed = (r-r.mean())/std
            else:
                normed = r-r.mean()
            for s, nr in zip(env_steps, normed): s.reward = float(nr) # write back
        return steps
