
import os, ray, random, csv
from typing import List, Dict, Optional, Tuple, Callable

# local imports
from unstable.core import Trajectory, Step, BaseTracker
from unstable.reward_transformations import ComposeFinalRewardTransforms, ComposeStepRewardTransforms, ComposeSamplingRewardTransforms
# from unstable.tracker import WandBTracker

# TODO doc-string

def write_eval_data_to_file(episode_info, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_info[0].keys()))
        writer.writeheader()
        writer.writerows(episode_info)

def write_training_data_to_file(batch, filename: str):
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pid', 'obs', 'act', 'reward', "env_id", "step_info"])  # header
        for step in batch:
            writer.writerow([step.pid, step.obs, step.act, step.reward, step.env_id, step.step_info])


@ray.remote
class StepBuffer:
    def __init__(
        self,
        max_buffer_size: int,
        final_reward_transformation: Optional[ComposeFinalRewardTransforms], 
        step_reward_transformation: Optional[ComposeStepRewardTransforms], 
        sampling_reward_transformation: Optional[ComposeSamplingRewardTransforms], 
        buffer_strategy: str = "random", # what happens when the buffer size exceeds 'max_buffer_size' many samples. Options are 'random' and 'sequential'
        tracker: Optional[BaseTracker] = None,
    ):
        self.max_buffer_size = max_buffer_size
        self.buffer_strategy = buffer_strategy 
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation

        self.steps: List[Step] = []
        self.training_steps = 0

        self.tracker = tracker

    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str):
        transformed_rewards = self.final_reward_transformation(trajectory.final_rewards, env_id=env_id) if self.final_reward_transformation is not None else trajectory.final_rewards
        n = len(trajectory.pid)
        for i in range(n): # these are already just steps by our model
            reward = transformed_rewards[trajectory.pid[i]]
            step_reward = self.step_reward_transformation(trajectory=trajectory, step_index=i, base_reward=reward) if self.step_reward_transformation is not None else reward
            self.steps.append(Step(
                pid=trajectory.pid[i], obs=trajectory.obs[i], act=trajectory.actions[i], reward=step_reward, env_id=env_id,
                step_info={"raw_reward": trajectory.final_rewards[trajectory.pid[i]], "transformed_end_of_game_reward": transformed_rewards[trajectory.pid[i]], "step_reward": step_reward}
            ))
        print(f"BUFFER SIZE: {len(self.steps)}, added {n} steps")

        excess_num_samples = int(len(self.steps) - self.max_buffer_size)
        if excess_num_samples > 0:
            randm_sampled = random.sample(self.steps, excess_num_samples)
            for b in randm_sampled:
                self.steps.remove(b)

    def get_batch(self, batch_size: int) -> List[Step]:
        batch = random.sample(self.steps, batch_size)
        for b in batch:
            self.steps.remove(b)
        batch = self.sampling_reward_transformation(batch) if self.sampling_reward_transformation is not None else batch
        try:
            if self.tracker:
                filename = os.path.join(ray.get(self.tracker.get_train_dir.remote()), f"train_data_step_{self.training_steps}.csv")
                write_training_data_to_file(batch=batch, filename=filename)
        except Exception as exc:
            print(f"EXCEPTION {exc}")
        # if self.args.log_training_data: # store training data as csv file
        #     filename = os.path.join(self.args.output_dir_train, f"train_data_step_{self.training_steps}.csv")
        #     write_training_data_to_file(batch=batch, filename=filename)
        self.training_steps += 1
        return batch

    def size(self) -> int:
        return len(self.steps)

    def clear(self):
        self.steps.clear()

