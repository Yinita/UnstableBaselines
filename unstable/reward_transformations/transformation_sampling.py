import numpy as np
from typing import List, Optional
from collections import defaultdict
from unstable.core import Step

class SamplingRewardTransform:
    def __call__(self, x: List[Step], env_id: Optional[str] = None) -> List[Step]: raise NotImplementedError

class ComposeSamplingRewardTransforms:
    def __init__(self, transforms: List[SamplingRewardTransform]):  self.transforms = transforms
    def __repr__(self):                                             return f"{self.__class__.__name__}({self.transforms})"
    def register(self, transform: SamplingRewardTransform):         self.transforms.append(transform)
    def __call__(self, x: List[Step]) -> List[Step]:
        for t in self.transforms:
            x = t(x)
        return x

class NormalizeRewards(SamplingRewardTransform):
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]:
        rewards = [step.reward for step in steps]
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-8  # avoid divide-by-zero

        for step in steps:
            step.reward = (step.reward - mean) #optionally: / std # TODO ablate
        return steps

class NormalizeRewardsByEnv(SamplingRewardTransform):
    def __init__(self, z_score: bool = False): self.z_score = z_score  # divide by std if True
    def __call__(self, steps: List[Step], env_id: Optional[str] = None) -> List[Step]:
        env_buckets = defaultdict(list)
        for step in steps: 
            env_buckets[step.env_id].append(step) # bucket by env
        for env_steps in env_buckets.values():
            r = np.asarray([s.reward for s in env_steps], dtype=np.float32)
            mean = r.mean()
            if self.z_score:    normed = (r - mean) / (r.std() + 1e-8)
            else:               normed = r - mean
            for s, nr in zip(env_steps, normed): # write back
                s.reward = float(nr)
        return steps

