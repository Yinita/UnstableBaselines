import random, itertools
import textarena as ta 
from typing import List, Dict, Tuple, Optional, Any, Callable

import ray 
from ray.util.actor_pool import ActorPool

# local imports
from unstable.core import Trajectory
from unstable.tracker import WandBTracker
from unstable.actors.vllm_actor import VLLMActor
from unstable.utils.templates import OBSERVATION_FORMATTING, ACTION_EXTRACTION


class CallableActorWrapper:
    def __init__(self, actor: VLLMActor, lora_path: str, obs_formatting_fn: Callable, action_extraction_fn: Callable):
        self.actor = actor 
        self.lora_path = lora_path
        self.obs_formatting_fn = obs_formatting_fn
        self.action_extraction_fn = action_extraction_fn

    def __call__(self, observation: str):
        formatted_prompt = self.obs_formatting_fn(observation=observation)
        raw_action = ray.get(self.actor.submit_prompt.remote(prompt=formatted_prompt, lora_path=self.lora_path))
        extracted_action, format_feedback = self.action_extraction_fn(raw_action=raw_action)
        return extracted_action #raw_action, extracted_action, format_feedback, formatted_prompt

    def get_full_response(self, observation: str):
        formatted_prompt = self.obs_formatting_fn(observation=observation)
        raw_action = ray.get(self.actor.submit_prompt.remote(prompt=formatted_prompt, lora_path=self.lora_path))
        extracted_action, format_feedback = self.action_extraction_fn(raw_action=raw_action)
        return raw_action, extracted_action, format_feedback, formatted_prompt

@ray.remote 
class Collector:
    def __init__(self, num_actors, step_buffer, model_pool, vllm_config: Dict[str, Any], training_envs: List[Tuple[str, int, Optional[str]]], tracker: Optional[WandBTracker] = None,):
        """
        TODO 
        vllm_config (Dict[str, Any]): # has to include the model_name and any other inference parameters you want
        """
        self.buffer = step_buffer 
        self.model_pool = model_pool 
        self.tracker = tracker 
        self.alive = True 
        self.training_envs = training_envs

        # build the actor pool
        actors = [VLLMActor.options(num_gpus=1).remote(vllm_config=vllm_config) for _ in range(num_actors)]
        self.actor_iter = itertools.cycle(actors)


    def _sample_env(self, seed=489):
        env_id, num_players, prompt_template = random.Random(seed).choice(self.training_envs)
        player_id = random.Random(seed+1).randrange(num_players)
        return env_id, num_players, player_id, prompt_template

    def _build_game_args(self, env_id, num_players, player_id, prompt_template, seed):
        actor = next(self.actor_iter)
        current_model_uid = ray.get(self.model_pool.latest_ckpt.remote())
        lora_path = ray.get(self.model_pool.ckpt_path.remote(current_model_uid))
        opponent_uid = ray.get(self.model_pool.sample.remote(uid_me=current_model_uid))
        opponent_path_or_name = ray.get(self.model_pool.ckpt_path.remote(opponent_uid))
        return dict(
            env_id=env_id, num_players=num_players, player_id=player_id, actor=actor, lora_path=lora_path, opponent_uid=opponent_uid, 
            model_uid=current_model_uid,
            opponent_path_or_name=opponent_path_or_name, prompt_template=prompt_template, seed=seed
        )
    
    
    def collect(self, num_workers: int):
        @ray.remote(num_cpus=0.1)
        def run_game(env_id, num_players, player_id, actor, lora_path, opponent_uid, model_uid, opponent_path_or_name, prompt_template, seed: int = 489):
            # build model and opponent
            # model, opponent = self._build_models(actor=actor, opponent_uid=opponent_uid, prompt_template=prompt_template)
            obs_formatting_fn = OBSERVATION_FORMATTING[prompt_template]; action_extraction_fn = ACTION_EXTRACTION[prompt_template]
            model = CallableActorWrapper(actor=actor, lora_path=lora_path, obs_formatting_fn=obs_formatting_fn, action_extraction_fn=action_extraction_fn)
            
            if opponent_uid is None: # mirror self-play
                opponent = model 
            elif opponent_uid.startswith("ckpt-"): # one of the previous checkpoints
                opponent = CallableActorWrapper(actor=actor, lora_path=opponent_path_or_name, obs_formatting_fn=obs_formatting_fn, action_extraction_fn=action_extraction_fn)
            else: # fixed opponent
                opponent = ta.agents.OpenRouterAgent(model_name=opponent_path_or_name)
        

            env = ta.make(env_id); env.reset(num_players=num_players, seed=seed); env.state.error_allowance = 0
            turn = 0

            traj = Trajectory()
            while True:
                pid, obs = env.get_observation()
                if pid==player_id:
                    raw_action, extracted_action, format_feedback, formatted_prompt = model.get_full_response(observation=obs)
                    done, info = env.step(action=extracted_action)
                    traj.pid.append(pid)
                    traj.obs.append(formatted_prompt)
                    traj.actions.append(raw_action)
                    format_feedback["invalid_move"] = 0 # change to 1 for final move if invalid
                    traj.format_feedbacks.append(format_feedback)

                else:
                    action = opponent(obs)
                    done, info = env.step(action=action)

                turn += 1
                if done: break

            traj.final_rewards = env.close(); traj.num_turns=turn
            
            return traj, player_id, env_id


        in_flight, iterated_seed = [], 0
        while self.alive:
            while len(in_flight) < num_workers: # top up queue
                env_id, num_players, player_id, prompt_template = self._sample_env(seed=iterated_seed)

                args = self._build_game_args(env_id, num_players, player_id, prompt_template, iterated_seed)
                future = run_game.remote(**args)

                in_flight.append((future, args["opponent_uid"], args["model_uid"]))
                iterated_seed += 1
                print(f"Num workers working: ", len(in_flight))

            # wait for one to finish
            done_ref, _ = ray.wait([f for f,_,_ in in_flight], num_returns=1)
            idx = next(i for i,(f,_,_) in enumerate(in_flight) if f==done_ref[0])
            future, opponent_uid, model_uid = in_flight.pop(idx)

            traj, player_id, env_id = ray.get(future)
            self.buffer.add_trajectory.remote(trajectory=traj, player_id=player_id, env_id=env_id)
            
            if opponent_uid is not None:
                self.model_pool.update_ratings.remote(uid_me=model_uid, uid_opp=opponent_uid, final_reward=traj.final_rewards[player_id])

            if self.tracker: # optionally log it
                self.tracker.add_trajectory.remote(trajectory=traj, player_id=player_id, env_id=env_id)

            print(f"Completed game")

    def stop(self): self.alive = False