
import os, ray, random
from typing import List, Dict, Optional, Tuple, Callable

# local imports
from core import Trajectory, Step
from utils.local_files import write_training_data_to_file



@ray.remote
class StepBuffer:
    def __init__(self, args, final_reward_transformation: Optional[Callable]=None, step_reward_transformation: Optional[Callable]=None, sampling_reward_transformation: Optional[Callable]=None):
        self.args = args
        self.final_reward_transformation = final_reward_transformation
        self.step_reward_transformation = step_reward_transformation
        self.sampling_reward_transformation = sampling_reward_transformation

        self.steps: List[Step] = []
        self.training_steps = 0

    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str):
        
        transformed_rewards = self.final_reward_transformation(trajectory.final_rewards, env_id=env_id) if self.final_reward_transformation is not None else trajectory.final_rewards
        n = len(trajectory.pid)
        for i in range(n): # these are already just steps by our model
            reward = transformed_rewards[trajectory.pid[i]]
            step_reward = self.step_reward_transformation(trajectory=trajectory, step_index=i, base_reward=reward) if self.step_reward_transformation is not None else reward
            self.steps.append(Step(pid=trajectory.pid[i], obs=trajectory.obs[i], act=trajectory.actions[i], reward=step_reward, env_id=env_id))
        print(f"BUFFER SIZE: {len(self.steps)}, added {n} steps")

        excess_num_samples = len(self.steps) - self.args.max_buffer_size
        if excess_num_samples > 0:
            randm_sampled = random.sample(self.steps, excess_num_samples)
            for b in randm_sampled:
                self.steps.remove(b)

    def get_batch(self, batch_size: int) -> List[Step]:
        batch = random.sample(self.steps, batch_size)
        for b in batch:
            self.steps.remove(b)
        batch = self.sampling_reward_transformation(batch) if self.sampling_reward_transformation is not None else batch
        if self.args.log_training_data: # store training data as csv file
            filename = os.path.join(self.args.output_dir_train, f"train_data_step_{self.training_steps}.csv")
            write_training_data_to_file(batch=batch, filename=filename)
        self.training_steps += 1
        return batch

    def size(self) -> int:
        return len(self.steps)

    def clear(self):
        self.steps.clear()

