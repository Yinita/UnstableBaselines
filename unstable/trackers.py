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
        self.metrics = {}

        if self.wandb_project is not None: # use wandb
            wandb.init(project=self.wandb_project, name=self.run_name)

        self._ma_range = 512; 
        self._tau = 0.001

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
            self._update_metric(name="Draw Rate", value=int(model_reward < opponent_reward), prefix=prefix, env_id=env_id)
            self._update_metric(name="Loss Rate", value=int(model_reward == opponent_reward), prefix=prefix, env_id=env_id)
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
        self.learner_metrics = info_dict

    def add_eval_episode(self, episode_info: Dict, final_reward: int, player_id: int, env_id: str, iteration: int, prefix: str="evaluation"):
        raise NotImplementedError



    def log_model_pool(self, iteration: int, match_counts, ts_dict: Dict[str, Dict[str, float]]): #, uid: str, mu: float, sigma: float, counts: Dict):
        pass



    # def log_trueskill(self, step, uid, mu, sigma):
    #     wandb.log({f"trueskill/{uid}/mu": mu, f"trueskill/{uid}/sigma": sigma, "learner/step": step})

    # def log_matchup_counts(self, step, counts: dict):
    #     # counts = {("ckpt-200","ckpt-195"): 17, ("ckpt-200","gemini"): 9, …}
    #     for (u1, u2), n in counts.items():
    #         wandb.log({f"matchups/{u1}_vs_{u2}": n, "learner/step": step})

    # def log_action_n_grams(self, action_n_grams: dict, env_id: str):
    #     def _normalize_ngrams(ngrams):
    #         total = sum(ngrams.values())
    #         return {k: v / total for k, v in ngrams.items()}
    #     def _compute_entropy(prob_dist):
    #         return -sum(p * math.log2(p) for p in prob_dist if p > 0)
    #     wandb_dict = {
    #         **{f"exploration ({env_id})/{n}-gram entropy": _compute_entropy(_normalize_ngrams(action_n_grams[n]).values()) for n in action_n_grams},
    #         **{f"exploration ({env_id})/unique {n}-grams": len(action_n_grams[n]) for n in action_n_grams}
    #         }
    #     wandb.log(wandb_dict)

    # def log_pairwise_n_gram_distances(self, dist_matrix: np.ndarray, uids: List[str]):
    #     wandb_dict = {}
    #     for i in range(dist_matrix.shape[0]):
    #         distances = [dist_matrix[i, j] for j in range(dist_matrix.shape[1]) if i != j]
    #         if distances:  # Only compute min if we have distances
    #             wandb_dict[f'opponent sampling/min dist.{uids[i]}'] = np.min(distances)
    #             wandb_dict[f'opponent sampling/max dist.{uids[i]}'] = np.max(distances)
    #         else:
    #             wandb_dict[f'opponent sampling/min dist.{uids[i]}'] = 0.0  # or np.nan
    #             wandb_dict[f'opponent sampling/max dist.{uids[i]}'] = 0.0  # or np.nan
    #     wandb.log(wandb_dict)

    # @staticmethod
    # def _entropy(counts: Dict[str, int]) -> float:
    #     total = sum(counts.values())
    #     return -sum((c / total) * math.log(c / total) for c in counts.values()) if total > 0.0 else 0.0