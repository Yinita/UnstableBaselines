import os, ray, wandb, datetime, collections
from typing import Dict, Optional
import numpy as np
# local imports
from unstable.core import BaseTracker, Trajectory


@ray.remote
class Tracker(BaseTracker):
    def __init__(self, run_name: str, accum_strategy: str="ma", output_dir: Optional[str]=None, wandb_project: Optional[str]=None):
        super().__init__(run_name=run_name, output_dir=output_dir)
        self.accum_strategy = accum_strategy
        self.wandb_project = wandb_project
        self._ma_range = 512; 
        self._tau = 0.001

        self.metrics = {}
        if self.wandb_project is not None: # use wandb
            wandb.init(project=self.wandb_project, name=self.run_name)


    def get_wandb_project(self):    return self.wandb_project
    def get_run_name(self):         return self.run_name

    def _log_metrics(self, prefix: str):
        if self.wandb_project: wandb.log({name: np.mean(value) if isinstance(value, collections.deque) else value for name, value in self.metrics.items() if prefix in name})

    def _update_metric(self, name: str, value: float|int|bool, prefix: str, env_id:str):
        for key in [f"{prefix}-{env_id}/{name}", f"{prefix}-all/{name}"]:
            if self.accum_strategy == "ma": 
                if key not in self.metrics: self.metrics[key] = collections.deque(maxlen=self._ma_range)
                self.metrics[key].append(value)

            elif self.accum_strategy =="ema": 
                self.metrics[key] = self.metrics.get(key) * (1-self._tau) + self._tau * value 

            else:
                raise NotImplementedError(f"The accumulation strategy {self.accum_strategy} is not implemented.")

    def _add_trajectory_to_metrics(self, trajectory: Trajectory, player_id: int, env_id: str, prefix: str="collection"):
        self._update_metric(name="Game Length", value=trajectory.num_turns, prefix=prefix, env_id=env_id) # log game length

        if len(trajectory.final_rewards) == 1: # single-player env
            self._update_metric(name="Reward", value=model_reward, prefix=prefix, env_id=env_id) # log all rewards

        elif len(trajectory.final_rewards) == 2: # two-player env
            model_reward = trajectory.final_rewards[player_id]
            opponent_reward = trajectory.final_rewards[1-player_id]
            self._update_metric(name="Win Rate", value=int(model_reward > opponent_reward), prefix=prefix, env_id=env_id)
            self._update_metric(name="Loss Rate", value=int(model_reward < opponent_reward), prefix=prefix, env_id=env_id)
            self._update_metric(name="Draw Rate", value=int(model_reward == opponent_reward), prefix=prefix, env_id=env_id)
            self._update_metric(name="Reward", value=model_reward, prefix=prefix, env_id=env_id) # log all rewards
            self._update_metric(name=f"Reward (pid={player_id})", value=model_reward, prefix=prefix, env_id=env_id) # log rewards by pid

        else:
            raise NotImplementedError("Tracker is not implemented for more than two player games")
        
        for i in range(len(trajectory.pid)): # iterate over turns
            if player_id == trajectory.pid[i]: # ensure we are only logging our own model here 
                for key in trajectory.format_feedbacks[i]: # log all format rewards 
                    self._update_metric(name=f"Format Success Rate - {key}", value=trajectory.format_feedbacks[i][key], prefix=prefix, env_id=env_id)

                # log obs and resp lengths
                self._update_metric(name="Response Length (avg. char)", value=len(trajectory.actions[i]), prefix=prefix, env_id=env_id)
                self._update_metric(name="Observation Length (avg. char)", value=len(trajectory.obs[i]), prefix=prefix, env_id=env_id)


    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str, prefix: str="collection"):
        self._add_trajectory_to_metrics(trajectory=trajectory, player_id=player_id, env_id=env_id, prefix=prefix)
        self._log_metrics(prefix=prefix)

    def log_learner(self, info_dict: Dict):
        self.metrics.update({f"learner/{name}": value for name, value in info_dict.items()})
        self._log_metrics(prefix="learner")
        # self.learner_metrics = info_dict
        # wandb.log()

    def _add_eval_episode_to_metrics(self, episode_info: Dict, final_rewards: Dict, player_id: int, env_id: str, iteration: int, prefix: str="evaluation"):
        self._update_metric(name="Game Length", value=len(episode_info), prefix=prefix, env_id=env_id) # log game length
        self._update_metric(name="Reward", value=final_rewards[player_id], prefix=prefix, env_id=env_id)

        if len(final_rewards) == 2: # two-player env
            model_reward = final_rewards[player_id]
            opponent_reward = final_rewards[1-player_id]
            self._update_metric(name="Win Rate", value=int(model_reward > opponent_reward), prefix=prefix, env_id=env_id)
            self._update_metric(name="Draw Rate", value=int(model_reward < opponent_reward), prefix=prefix, env_id=env_id)
            self._update_metric(name="Loss Rate", value=int(model_reward == opponent_reward), prefix=prefix, env_id=env_id)

            # save csv 
            # TODO

        else:
            raise NotImplementedError("Tracker is not implemented for more than two player games")

    def add_eval_episode(self, episode_info: Dict, final_rewards: Dict, player_id: int, env_id: str, iteration: int, prefix: str="evaluation"):
        self._add_eval_episode_to_metrics(episode_info=episode_info, final_rewards=final_rewards, player_id=player_id, env_id=env_id, iteration=iteration, prefix=prefix)
        self._log_metrics(prefix=prefix)


    def log_model_pool(self, iteration: int, match_counts, ts_dict: Dict[str, Dict[str, float]], exploration_ratios: Dict[str, float]): #, uid: str, mu: float, sigma: float, counts: Dict):
        log_dict = {}

        # Log TrueSkill values
        for uid, ts in ts_dict.items():
            log_dict[f"trueskill/{uid}/mu"] = ts["mu"]
            log_dict[f"trueskill/{uid}/sigma"] = ts["sigma"]

        # Log match counts
        for (uid1, uid2), count in match_counts.items():
            log_dict[f"matchups/{uid1}_vs_{uid2}"] = count

        # Log n-gram exploration stats (if provided)
        # if exploration_ratios:
        #     for (uid1, uid2), ratios in exploration_ratios.items():
        #         for ngram_type, ratio in ratios.items():
        #             log_dict[f"{f"exploration/{uid1}_vs_{uid2}"}/{ngram_type}_diversity"] = ratio
        for name, count in exploration_ratios.items():
            log_dict[f"exploration/{name}"] = count

        # Include iteration step for context
        # log_dict["learner/step"] = iteration

        # Push to WandB
        if self.wandb_project:
            wandb.log(log_dict)

