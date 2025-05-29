import time, random, threading, traceback
from typing import List, Tuple, Dict, Optional

import ray
import textarena as ta 

# local imports
from unstable.core import Trajectory
from unstable.trajectory_buffer import StepBuffer
from unstable.utils.templates import OBSERVATION_FORMATTING, ACTION_EXTRACTION


@ray.remote
class Collector:
    def __init__(
        self,
        num_actors: int = 1,
        buffer: StepBuffer, 
        tracker, # TODO
        training_envs: List[Tuple[str, int, Optional[str]]], # env-id, num_players, observation_template (optional)
        evaluation_envs: List[Tuple[str, int, Optional[str]]], # env-id, num_players, observation_template (optional)
        num_collection_workers: int = 384,
        num_evaluation_workers: int = 32,
        opponent_strategy: str = "self-play", # either "self-play", "fixed", "performance-sampling"
        opponent_selection
        vllm_dict: Dict[str, Any] = None, # vllm generation parameters
        num_evaluation_games: int = 64,
        evaluate_every_n_checkpoints: int = 5
        initial_lora_path: Optional[str] = None,
        eval_opponent_name: str = "google/gemini-2.0-flash-lite-001", # by default use gemini-2.0-flash-lite
        fixed_opponent_names: List[str] = ["google/gemini-2.0-flash-lite-001"],
        self_play_lag_range: Tuple[int, int] = (1, 7)
    ): 
        self.buffer = buffer
        self.tracker = tracker 

        self.opponent_strategy = opponent_strategy
        self.fixed_opponent_names = fixed_opponent_names
        self.self_play_lag_range = self_play_lag_range 

        self.evaluate_every_n_checkpoints = evaluate_every_n_checkpoints
        self.num_evaluation_games = num_evaluation_games
        self.training_envs = [(env_id, num_players, template if len(t) > 2 else None) for t in training_envs for env_id, num_players, *template in [t]]
        self.evaluation_envs = [(env_id, num_players, template if len(t) > 2 else None) for t in evaluation_envs for env_id, num_players, *template in [t]]
        self.num_collection_workers = num_collection_workers
        self.num_evaluation_workers = num_evaluation_workers
        self.eval_opponent_name = eval_opponent_name

        self.actor_group: List[ray.actor.ActorHandle] = []
        self.lora_paths: List[Optional[str]] = [initial_lora_path] # either None or the actual path
        self.checkpoints_to_evaluate: List[Tuple[Optional[str], int, Dict[str, int]]] = [(self.lora_paths[0], 0, {})] # path (str), iteration (int), env eval acount (Dict[str, int])
        self._done = False 
        self.actor_group = [RayActor.remote(vllm_dict) for _ in range(num_actors)]

    def mark_done(self): 
        self._done = True 

    def _clean_collection_futures(self):
        ready, not_ready = ray.wait(self.collection_outstanding, timeout=0, num_returns=len(self.collection_outstanding))
        for obj_ref in ready:
            try: 
                env, traj, done, info, player_id, ending_pid = ray.get(obj_ref) # TODO adjust to work with GRPO. For now, just always expect done=True
                traj.final_rewards = env.close()
                if info["end_by_invalid"] and player_id==ending_pid:
                    traj.format_feedbacks[-1]["invalid_move"] = 1
                traj.num_turns = info["turn_count"]
                print(f"GAME FINISHED< ADDING TO BUFFER. num turns: {traj.num_turns} [steps by our model: {len(traj.pid)}]") # TODO tmp
                ray.get(self.buffer.add_trajectory.remote(traj, player_id=player_id, env_id=env.env_id))
                ray.get(self.tracker.add_trajectory.remote(traj, player_id=player_id, env_id=env.env_id))

            except Exception as e: 
                print(f"[FUTURE ERROR]: {e}", traceback.format_exc())
        self.collection_outstanding = list(not_ready)

    def _clean_evaluation_futures(self):
        ready, not_ready = ray.wait(self.evaluation_outstanding, timeout=0, num_returns=len(self.evaluation_outstanding))
        for obj_ref in ready:
            try: 
                episode_info, final_rewards, player_id, env_id, ckpt_iteration = ray.get(obj_ref)
                ray.get(self.tracker.add_eval_episode.remote(episode_info, final_rewards, player_id, env_id, ckpt_iteration))
            except Exception as e: 
                print(f"[FUTURE ERROR]: {e}", traceback.format_exc())
        self.evaluation_outstanding = list(not_ready)
    
    def _sample_lora(self): # get random lora weights from within the opponent delay window
        lower, upper = self.self_play_lag_range
        return random.choice(self.lora_paths[-min(upper, len(self.lora_paths)):-lower]) if len(self.lora_paths) > lower else self.lora_paths[0] 

    def _sample_environment(self, seed, _type="train"):
        env_id, num_players, prompt_template = random.Random(seed=seed).choice(self.training_envs if _type=="train" else self.evaluation_envs)
        player_id = random.Random(seed=seed).randint(0, num_players)
        return env_id, num_players, player_id, OBSERVATION_FORMATTING[prompt_template], ACTION_EXTRACTION[prompt_template]

    def collect(self):
        iter_seed = 0
        self.collection_outstanding = []
        self.evaluation_outstanding = []

        def build_env(env_id, num_players, seed):
            env = ta.make(env_id=env_id)
            env.reset(num_player=num_players, seed=seed)
            env.error_allowance = 0     
            return env 

        def loop():
            while True:
                if self._done: print("[ACTOR LOOP] Training is done. Exiting actor loop."); break # check if we are done with everything

                # clean up finished futures
                self._clean_collection_futures()
                self._clean_evaluation_futures()

                # replenish collection futures
                if len(self.collection_outstanding) < self.num_collection_workers:
                    # get environment
                    env_id, num_players, player_id, obs_formatting_fn, action_extraction_fn = self._sample_environment(seed=iter_seed, _type="train")
                    current_model = CallableActorWrapper(actor=random.choice(self.actor_group), lora_path=self.lora_paths[-1], obs_formatting_fn=obs_formatting_fn, action_extraction_fn=action_extraction_fn)

                    # get opponent
                    if self.opponent_strategy == "self-play":
                        opponent = CallableActorWrapper(actor=random.choice(self.actor_group), lora_path=self.lora_paths[-1], obs_formatting_fn=obs_formatting_fn, action_extraction_fn=action_extraction_fn)
                    elif self.opponent_strategy = "self-play-sampling":
                        opponent = CallableActorWrapper(actor=random.choice(self.actor_group), lora_path=self._sample_lora(), obs_formatting_fn=obs_formatting_fn, action_extraction_fn=action_extraction_fn)
                    elif self.opponent_strategy == "fixed":
                        opponent = ta.agents.OpenRouterAgent(model_name=random.choice(self.fixed_opponent_names))
                    elif self.opponent_strategy == "performance-sampling":
                        raise NotImplementedError   
                    else:
                        raise NotImplementedError   

                    # build the env and Trajectory 
                    env = build_env(env_id=env_id, num_players=num_players, seed=iter_seed)
                    traj = Trajectory()

                    future = run_game.remote(env=env, player_id=player_id, current_model=current_model, traj=traj)
                    self.collection_outstanding.append(future)
                    iter_seed += 1

                # replenish evaluation futures
                elif len(self.evaluation_outstanding) < self.num_evaluation_workers:
                    env_id, num_players, player_id, obs_formatting_fn, action_extraction_fn = self._sample_environment(seed=iter_seed, _type="eval")
                    # check if we have something to evaluate
                    run_eval, lora_path, ckpt_iteration = self._get_checkpoint_to_evaluate(env_id=env_id)
                    if run_eval:
                        current_model = CallableActorWrapper(actor=random.choice(self.actor_group), lora_path=self.lora_paths[-1], obs_formatting_fn=obs_formatting_fn, action_extraction_fn=action_extraction_fn)
                        opponent = ta.agents.OpenRouterAgent(model_name=self.eval_opponent_name)
                    
                    # build the env and Trajectory
                    env = build_env(env_id=env_id, num_players=num_players, seed=iter_seed)
                    traj = Trajectory()

                    future = run_eval_game.remote(env=env, player_id=player_id, current_model=current_model, ckpt_iteration=ckpt_iteration)
                    self.evaluation_outstanding.append(future)

                else:
                    time.sleep(0.05)


        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

    def add_new_lora_paths(self, new_path: str, iteration: int):
        self.lora_paths.append(new_path) # add to collection checkpoitns
        if iteration % self.evaluate_every_n_checkpoints==0: # add every n-th lora path for evaluation
            self.checkpoints_to_evaluate.append((new_path, iteration, {})) 

    def _get_checkpoint_to_evaluate(self, env_id: str): # TODO make more efficient
        # check if for this env id we already ran all games for that checkpoint
        for i in range(len(self.checkpoints_to_evaluate)):
            # check if viable
            num_evals_run = self.checkpoints_to_evaluate[i][2].get(env_id, 0)
            if num_evals_run < self.num_evaluation_games:
                # increment and return
                self.checkpoints_to_evaluate[i][2][env_id] = self.checkpoints_to_evaluate[i][2].get(env_id, 0) + 1
                return True, self.checkpoints_to_evaluate[i][0], self.checkpoints_to_evaluate[i][1]
        return False, None, None # None were found


@ray.remote
def run_game(env, player_id, current_model, opponent, traj, num_steps: Optional[int] = None):
    done, steps = False, 0 
    while not done and (num_steps is None or steps < num_steps):
        pid, obs = env.get_observation()

        if pid == player_id: # current model moves
            raw_action, extracted_action, format_feedback, formatted_prompt = current_model.get_full_response(observation=obs)
            done, info = env.step(action=extracted_action)

            # add to trajectory
            traj.pid.append(pid)
            traj.obs.append(formatted_prompt)
            traj.actions.append(action)
            format_feedback["invalid_move"] = 0 # change to 1 for final move if invalid
            traj.format_feedbacks.append(format_feedback)

        else: # sample action from opponent
            action = opponent(observation)
            done, info = env.step(action=action)

    return env, traj, done, info, player_id, pid

@ray.remote
def run_eval_game(env, player_id, current_model, opponent, ckpt_iteration):
    episode_info = []
    turns = 0
    done = False
    while not done:
        pid, obs = env.get_observation()

        if pid == player_id: # current model moves
            full_action, action, format_feedback, formatted_prompt = current_model.get_full_response(observation=obs)
            model_name = "current_ckpt"
        else: # sample action from opponent
            full_action = action = opponent(observation)
            done, info = env.step(action=action)
            model_name = opponent.model_name

        done, info = env.step(action=action)
        episode_info.append({
            "pid": pid, "model_name": model_name, "observation": obs, "full_action": full_action, 
            "submitted_action": action, "done": done, "info": info, "step": turns
        })
        turns += 1
    
    return episode_info, env.close(), player_id, env.env_id, ckpt_iteration

