
import os, ray, csv, random, logging
from typing import List, Dict, Optional, Tuple, Callable

# local imports
from unstable.core import Trajectory, Step, BaseTracker
from unstable.utils.logging import setup_logger
from unstable.reward_transformations import ComposeFinalRewardTransforms, ComposeStepRewardTransforms, ComposeSamplingRewardTransforms


# TODO doc-strings

def write_eval_data_to_file(episode_info, filename):
    """Dump a list of dicts (one row per turn) to CSV `filename`."""
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_info[0].keys()))
        writer.writeheader()
        writer.writerows(episode_info)

def write_training_data_to_file(batch, filename: str):
    """Save a Step mini-batch as CSV for offline inspection."""
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pid', 'obs', 'act', 'reward', "env_id", "step_info"])  # header
        for step in batch: writer.writerow([step.pid, step.obs, step.act, step.reward, step.env_id, step.step_info])

@ray.remote
class StepBuffer:
    """
    In-memory replay buffer that stores *Steps* until the learner samples them.
    Supports optional reward-transformation hooks executed **on read** and/or **on write**.

    Parameters
    ----------
    max_buffer_size (int): Hard cap on stored Steps. Older entries are dropped according to `buffer_strategy`.
    final_reward_transformation, step_reward_transformation, sampling_reward_transformation Callables composed via the `reward_transformations.*` utilities.
    buffer_strategy {"random", "sequential"}: How to evict when over capacity.
    tracker (ray.ActorHandle or None): For auto-export of every sampled batch to CSV.
    """
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
        self.collect = True

        self.steps: List[Step] = []
        self.training_steps = 0

        self.tracker = tracker
        self.local_storage_dir = ray.get(self.tracker.get_train_dir.remote())

        # setup logging
        log_dir = ray.get(tracker.get_log_dir.remote())
        self.logger = setup_logger("step_buffer", log_dir)

    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str):
        """ Decompose a finished trajectory into individual 'Step's and append. """
        # transformed_rewards = self.final_reward_transformation(trajectory.final_rewards, env_id=env_id) if self.final_reward_transformation is not None else trajectory.final_rewards
        try:
            transformed_rewards = self.final_reward_transformation(trajectory.final_rewards, env_id=env_id) if self.final_reward_transformation else trajectory.final_rewards
        except Exception as exc: self.logger.exception(f"reward-transformation blew up  env={env_id} -\n\n{exc}\n\n"); raise
        
        n = len(trajectory.pid)
        for i in range(n): # these are already just steps by our model
            reward = transformed_rewards[trajectory.pid[i]]
            step_reward = self.step_reward_transformation(trajectory=trajectory, step_index=i, base_reward=reward) if self.step_reward_transformation is not None else reward
            step_info = {"raw_reward": trajectory.final_rewards[trajectory.pid[i]], "transformed_end_of_game_reward": transformed_rewards[trajectory.pid[i]], "step_reward": step_reward}
            self.steps.append(Step(pid=trajectory.pid[i], obs=trajectory.obs[i], act=trajectory.actions[i], reward=step_reward, env_id=env_id, step_info=step_info))
        self.logger.info(f"BUFFER SIZE: {len(self.steps)}, added {n} steps")

        # excess_num_samples = int(len(self.steps) - self.max_buffer_size)
        excess_num_samples = max(0, len(self.steps) - self.max_buffer_size)
        self.logger.info(f"Excess Num Samples: {excess_num_samples}")
        if excess_num_samples > 0:
            self.logger.info(f"Downsampling buffer because of excess samples")
            randm_sampled = random.sample(self.steps, excess_num_samples)
            for b in randm_sampled:
                self.steps.remove(b)
            self.logger.info(f"Buffer size after downsampling: {len(self.steps)}")

    def get_batch(self, batch_size: int) -> List[Step]:
        """
        Random-sample a mini-batch and *remove* it from the buffer.
        On each call the batch is optionally reweighted by `sampling_reward_transformation`.
        """
        try:                        batch = random.sample(self.steps, batch_size)
        except ValueError as exc:   self.logger.error("requested %s samples - only %s in buffer", batch_size, len(self.steps)); raise
        for b in batch:
            self.steps.remove(b)
        batch = self.sampling_reward_transformation(batch) if self.sampling_reward_transformation is not None else batch
        self.logger.info(f"Sampling {len(batch)} samples from buffer.")
        try:
            if self.tracker:
                filename = os.path.join(self.local_storage_dir, f"train_data_step_{self.training_steps}.csv")
                write_training_data_to_file(batch=batch, filename=filename)
        except Exception as exc:
            print(f"EXCEPTION {exc}")
        self.training_steps += 1
        self.logger.info(f"returning batch")
        return batch

    def size(self) -> int:
        self.logger.info(f"Getting size: {len(self.steps)}")
        return len(self.steps)
    def clear(self):        self.steps.clear()
    def stop(self):         self.collect = False
    def continue_collection(self):      return self.collect

