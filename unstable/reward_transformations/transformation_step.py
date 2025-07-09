from typing import List
from unstable.types import PlayerTrajectory

class StepRewardTransform:
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float: raise NotImplementedError

class ComposeStepRewardTransforms:
    def __init__(self, transforms: List[StepRewardTransform]): self.transforms = transforms
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        for transform in self.transforms: reward = transform(player_traj, step_index, reward)
        return reward

class RewardForFormat(StepRewardTransform):
    def __init__(self, reward: float=0, penalty: float=0): self.reward, self.penalty = reward, penalty
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        reward += (self.reward if player_traj.format_feedbacks[step_index].get("correct_answer_format") else self.penalty)
        return reward

class PenaltyForInvalidMove(StepRewardTransform):
    def __init__(self, reward: float=0, penalty: float=0): self.reward, self.penalty = reward, penalty
    def __call__(self, player_traj: PlayerTrajectory, step_index: int, reward: float) -> float:
        reward += (self.penalty if player_traj.format_feedbacks[step_index].get("invalid_move") else self.reward)
        return reward