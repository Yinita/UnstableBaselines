import os, ray, wandb, collections
from typing import Optional, Union
import numpy as np
from unstable.core import BaseTracker, Trajectory


@ray.remote
class Tracker(BaseTracker):
    COLLECT_FLUSH_EVERY: int = 256 # How often to push collection stats (in *trajectories*)
    TOPK: int = 10 # Only keep the K most‑played match‑ups in the dashboard table

    def __init__(self, run_name: str, accum_strategy: str="ma", output_dir: Optional[str]=None, wandb_project: Optional[str]=None) -> None:
        super().__init__(run_name=run_name, output_dir=output_dir)
        self.accum_strategy = accum_strategy
        self.wandb_project = wandb_project
        self._ma_range = 512; self._tau = 0.001

        # metric storage
        self._metrics: dict[str, Union[collections.deque, float]] = {}
        self._buffer: dict[str, Union[int, float, bool, "wandb.Table"]] = {}
        self._collect_counter: int = 0  # trajectories since last flush
        self._inference: dict[str, dict[str, float]] = {}

        if self.wandb_project is not None:
            wandb.init(project=self.wandb_project, name=run_name)
            wandb.define_metric("learner/step", step_metric="learner/step")
            wandb.define_metric("env/episode", step_metric="env/episode")

    def _aggregate(self, prefix: str) -> dict[str, Union[int, float]]:
        """Collapse internal metric deques/emas into scalars for prefix """
        out: dict[str, Union[int, float]] = {}
        for name, val in self._metrics.items():
            if not name.startswith(prefix): 
                continue
            out[name] = (float(np.mean(val)) if val else 0.0) if isinstance(val, collections.deque) else val
        return out

    def _flush(self, force: bool = False):
        if (force or self._collect_counter >= self.COLLECT_FLUSH_EVERY) and self.wandb_project and self._buffer:
            try:
                wandb.log(self._buffer)
            except Exception as e:
                # never crash the actor on WANDB issues
                logger.error("wandb.log failed: %s", e)
        self._buffer.clear()
        self._collect_counter = 0

    def get_wandb_project(self):    return self.wandb_project
    def get_run_name(self):         return self.run_name

    # metric bookkeeping
    def _update_metric(self, name: str, value: float | int | bool, prefix: str, env_id: str):
        for key in (f"{prefix}-{env_id}/{name}", f"{prefix}-all/{name}"):
            match self.accum_strategy:
                case "ma":  self._metrics.setdefault(key, collections.deque(maxlen=self._ma_range)).append(value)
                case "ema": self._metrics[key] = self._metrics.get(key, 0.0) * (1 - self._tau) + self._tau * value
                case _:     raise NotImplementedError

    def _batch_update_metrics(self, pairs: list[tuple[str, float|int|bool]], prefix: str, env_id: str) -> None:
        for n, v in pairs:  
            self._update_metric(n, v, prefix, env_id)

    # trajectory collection
    def _add_trajectory_to_metrics(self, trajectory: Trajectory, player_id: int, env_id: str, prefix: str="collection") -> None:
        m_reward, o_reward = trajectory.final_rewards[player_id], trajectory.final_rewards[1 - player_id]
        pairs = [("Game Length", trajectory.num_turns)]
        match len(trajectory.final_rewards):
            case 1: pairs.append(("Reward", m_reward))
            case 2: pairs.extend([("Win Rate", int(m_reward > o_reward)), ("Loss Rate", int(m_reward < o_reward)), ("Draw Rate", int(m_reward == o_reward)), ("Reward", m_reward), (f"Reward (pid={player_id})", m_reward)])
            case _: raise NotImplementedError

        for i, pid in enumerate(trajectory.pid):
            if pid != player_id: continue
            for k, v in trajectory.format_feedbacks[i].items():
                pairs.append((f"Format Success Rate - {k}", v))
            pairs.extend([("Response Length (avg. char)", len(trajectory.actions[i])), ("Observation Length (avg. char)", len(trajectory.obs[i]))])
        self._batch_update_metrics(pairs, prefix, env_id)

    def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str, prefix: str = "collection"):
        self._add_trajectory_to_metrics(trajectory, player_id, env_id, prefix)
        self._collect_counter += 1
        if self._collect_counter % self.COLLECT_FLUSH_EVERY == 0:
            self._buffer.update(self._aggregate(prefix))
            self._flush()

    # learner updates
    def log_learner(self, info: dict):
        self._metrics.update({f"learner/{k}": v for k, v in info.items()})
        self._buffer.update(self._aggregate("learner"))

    # evaluation episodes
    def _add_eval_episode_to_metrics(self, episode_info: dict, final_rewards: dict, player_id: int, env_id: str, prefix: str="evaluation") -> None:
        pairs = [("Game Length", len(episode_info)), ("Reward", final_rewards[player_id])]
        if len(final_rewards) == 2:
            m_reward, o_reward = final_rewards[player_id], final_rewards[1 - player_id]
            pairs.extend([("Win Rate", int(m_reward > o_reward)), ("Loss Rate", int(m_reward < o_reward)), ("Draw Rate", int(m_reward == o_reward)), ("Reward", m_reward), (f"Reward (pid={player_id})", m_reward)])
        elif len(final_rewards) > 2: raise NotImplementedError
        self._batch_update_metrics(pairs, prefix, env_id)

    def add_eval_episode(self, episode_info: dict, final_rewards: dict, player_id: int, env_id: str, iteration: int, prefix: str="evaluation") -> None:
        self._add_eval_episode_to_metrics(episode_info, final_rewards, player_id, env_id, prefix)
        self._buffer.update(self._aggregate(prefix))
        self._flush()

    # model‑pool snapshots
    def log_model_pool(self, step: int, match_counts: dict, ts_dict: dict, exploration: dict) -> None:
        # latest checkpoint = highest integer suffix
        cur = max((uid for uid in ts_dict if uid.startswith("ckpt-")), key=lambda u: int(u.split("-")[1]), default=None)
        if cur is not None:
            self._buffer["trueskill/mu"] = ts_dict[cur]["mu"]
            self._buffer["trueskill/sigma"] = ts_dict[cur]["sigma"]

        # top‑K match‑ups table
        top = sorted(match_counts.items(), key=lambda kv: kv[1], reverse=True)[: self.TOPK]
        table = wandb.Table(columns=["uid_a", "uid_b", "games"], data=[[*pair, cnt] for pair, cnt in top])
        self._buffer["pool/top_matchups"] = table

        # exploration metrics
        for n, c in exploration.items():
            self._buffer[f"pool/exploration/{n}"] = c
        self._flush(force=True)

    # tracker.py
    def get_latest_learner_metrics(self):
        # pull from _metrics; safe default values
        return {k.split('/',1)[1]: v for k,v in self._buffer.items() if k.startswith('learner/')}

    def log_inference(self, actor: str, gpu_ids, stats: dict[str, dict[str, float]]):
        self._inference[actor] = stats

        # Optionally log to wandb
        if self.wandb_project:
            flat_stats = {}
            for lora, values in stats.items():
                for key, val in values.items():
                    flat_stats[f"inference/{lora}/{key}"] = val
            flat_stats["inference/actor"] = actor
            self._buffer.update(flat_stats)
            self._flush()

    def get_latest_inference_metrics(self) -> dict[str, dict[str, float]]:
        return self._inference


    # def log_inference(self, actor: str, stats: dict[str, dict[str, float]]):
    #     """
    #     `stats` comes straight from each actor:
    #         {lora_name: {"queue": .., "running": .., "tok_s": ..}}
    #     We simply *add* counts across actors and keep the latest rate.
    #     """
    #     for lora, meta in stats.items():
    #         agg = self._inference.setdefault(lora, {"queue": 0, "running": 0, "tok_s": 0.0})
    #         agg["queue"]   += meta["queue"]
    #         agg["running"] += meta["running"]
    #         agg["tok_s"]    = meta["tok_s"]          # last writer wins  (≈total)

    #     # ✅  Also drop them into the W&B buffer so you get a free plot
    #     for lora, meta in self._inference.items():
    #         prefix = f"inference/{lora}"
    #         self._buffer[f"{prefix}/queue"]   = meta["queue"]
    #         self._buffer[f"{prefix}/running"] = meta["running"]
    #         self._buffer[f"{prefix}/tok_s"]   = meta["tok_s"]

    # # ---------------- NEW: getter ----------------------
    # def get_latest_inference_metrics(self):
    #     # return a *copy* so the dashboard can mutate safely
    #     return {k: v.copy() for k, v in self._inference.items()}




# import os, ray, wandb, datetime, collections
# from typing import List, Tuple, Dict, Optional
# import numpy as np

# # local imports
# from unstable.core import BaseTracker, Trajectory


# @ray.remote
# class Tracker(BaseTracker):
#     def __init__(self, run_name: str, accum_strategy: str="ma", output_dir: Optional[str]=None, wandb_project: Optional[str]=None):
#         super().__init__(run_name=run_name, output_dir=output_dir)
#         self.accum_strategy = accum_strategy
#         self.wandb_project = wandb_project
#         self._ma_range = 512; self._tau = 0.001

#         self.metrics = {}
#         if self.wandb_project is not None: wandb.init(project=self.wandb_project, name=self.run_name) # use wandb


#     def get_wandb_project(self):    return self.wandb_project
#     def get_run_name(self):         return self.run_name

#     def _log_metrics(self, prefix: str):
#         if self.wandb_project: wandb.log({name: np.mean(value) if isinstance(value, collections.deque) else value for name, value in self.metrics.items() if prefix in name})

#     def _update_metric(self, name: str, value: float|int|bool, prefix: str, env_id:str):
#         for key in [f"{prefix}-{env_id}/{name}", f"{prefix}-all/{name}"]:
#             match self.accum_strategy:
#                 case "ma":  self.metrics.setdefault(key, collections.deque(maxlen=self._ma_range)).append(value)
#                 case "ema": self.metrics[key] = self.metrics.get(key, 0.0) * (1-self._tau) + self._tau * value 
#                 case _:     raise NotImplementedError(f"The accumulation strategy {self.accum_strategy} is not implemented.")
    
#     def _batch_update_metrics(self, name_value_pairs: List[Tuple[str, float|int|bool]], prefix: str, env_id: str):
#         for name, value in name_value_pairs: 
#             self._update_metric(name=name, value=value, prefix=prefix, env_id=env_id)

#     def _add_trajectory_to_metrics(self, trajectory: Trajectory, player_id: int, env_id: str, prefix: str="collection"):
#         m_reward = trajectory.final_rewards[player_id]; o_reward = trajectory.final_rewards[1-player_id]
#         name_value_pairs = [("Game Length", trajectory.num_turns)] # log game length
#         match len(trajectory.final_rewards):
#             case 1: name_value_pairs.extend([("Reward", m_reward)]) # single-player env
#             case 2: name_value_pairs.extend([("Win Rate", int(m_reward>o_reward)), ("Loss Rate", int(m_reward<o_reward)), ("Draw Rate", int(m_reward==o_reward)), ("Reward", m_reward), (f"Reward (pid={player_id})", m_reward)]) # two-player env
#             case _: raise NotImplementedError("Tracker is not implemented for more than two player games")
#         for i in range(len(trajectory.pid)): # iterate over turns
#             if player_id == trajectory.pid[i]: # ensure we are only logging our own model here 
#                 for key in trajectory.format_feedbacks[i]: name_value_pairs.append((f"Format Success Rate - {key}", trajectory.format_feedbacks[i][key])) # log all format rewards 
#                 name_value_pairs.extend([("Response Length (avg. char)", len(trajectory.actions[i])), ("Observation Length (avg. char)", len(trajectory.obs[i]))]) # log obs and resp lengths
#         self._batch_update_metrics(name_value_pairs=name_value_pairs, prefix=prefix, env_id=env_id)

#     def add_trajectory(self, trajectory: Trajectory, player_id: int, env_id: str, prefix: str="collection"):
#         self._add_trajectory_to_metrics(trajectory=trajectory, player_id=player_id, env_id=env_id, prefix=prefix)
#         self._log_metrics(prefix=prefix)

#     def log_learner(self, info_dict: Dict):
#         self.metrics.update({f"learner/{name}": value for name, value in info_dict.items()})
#         self._log_metrics(prefix="learner")
#         # self.learner_metrics = info_dict
#         # wandb.log()

#     def _add_eval_episode_to_metrics(self, episode_info: Dict, final_rewards: Dict, player_id: int, env_id: str, iteration: int, prefix: str="evaluation"):
#         name_value_pairs = [("Game Length", len(episode_info)), ("Reward", final_rewards[player_id])] # Log game length and reward
#         if len(final_rewards) == 2: # two-player env
#             m_reward=final_rewards[player_id]; o_reward=final_rewards[1-player_id]
#             name_value_pairs.extend([("Win Rate", int(m_reward>o_reward)), ("Loss Rate", int(m_reward<o_reward)), ("Draw Rate", int(m_reward==o_reward)), ("Reward", m_reward), (f"Reward (pid={player_id})", m_reward)])
#             # save csv 
#             # TODO
#         elif len(final_rewards) > 2: raise NotImplementedError("Tracker is not implemented for more than two player games")

#         self._batch_update_metrics(name_value_pairs=name_value_pairs, prefix=prefix, env_id=env_id)


#     def add_eval_episode(self, episode_info: Dict, final_rewards: Dict, player_id: int, env_id: str, iteration: int, prefix: str="evaluation"):
#         self._add_eval_episode_to_metrics(episode_info=episode_info, final_rewards=final_rewards, player_id=player_id, env_id=env_id, iteration=iteration, prefix=prefix)
#         self._log_metrics(prefix=prefix)


#     # def log_model_pool(self, iteration: int, match_counts, ts_dict: Dict[str, Dict[str, float]], exploration_ratios: Dict[str, float]): #, uid: str, mu: float, sigma: float, counts: Dict):
#     #     log_dict = {}

#     #     # Log TrueSkill values
#     #     for uid, ts in ts_dict.items():
#     #         log_dict[f"trueskill/{uid}/mu"] = ts["mu"]
#     #         log_dict[f"trueskill/{uid}/sigma"] = ts["sigma"]

#     #     # Log match counts
#     #     for (uid1, uid2), count in match_counts.items():
#     #         log_dict[f"matchups/{uid1}_vs_{uid2}"] = count

#     #     # Log n-gram exploration stats (if provided)
#     #     # if exploration_ratios:
#     #     #     for (uid1, uid2), ratios in exploration_ratios.items():
#     #     #         for ngram_type, ratio in ratios.items():
#     #     #             log_dict[f"{f"exploration/{uid1}_vs_{uid2}"}/{ngram_type}_diversity"] = ratio
#     #     for name, count in exploration_ratios.items():
#     #         log_dict[f"exploration/{name}"] = count

#     #     # Include iteration step for context
#     #     # log_dict["learner/step"] = iteration

#     #     # Push to WandB
#     #     if self.wandb_project:
#     #         wandb.log(log_dict)


#     def log_model_pool(self, step: int, match_counts: dict, ts_dict: dict, exploration: dict):
#         cur = max(ts_dict, key=lambda uid: ts_dict[uid]["mu"])  # choose the latest or best
#         self._buffer["trueskill/mu"] = ts_dict[cur]["mu"]
#         self._buffer["trueskill/sigma"] = ts_dict[cur]["sigma"]

#         # top-K matchups table
#         top = sorted(match_counts.items(), key=lambda kv: kv[1], reverse=True)[:self.TOPK]
#         table = wandb.Table(columns=["uid_a", "uid_b", "games"], data=[[*pair, cnt] for pair, cnt in top])
#         self._buffer["pool/top_matchups"] = table

#         # flush immediately (changes infrequently)
#         wandb.log(self._buffer)
#         self._buffer.clear()
