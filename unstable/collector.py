import re, random, logging, itertools
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Protocol, Optional

import ray
from ray.exceptions import RayActorError, RayTaskError

import textarena as ta
assert ta.__version__ >= "0.6.16", f"TextArena package version is too old: {ta.__version__}. Required version is at least 0.6.16."

# local imports
from unstable.actor import VLLMActor
from unstable._types import GameSpec, GameInformation, PlayerTrajectory, TaskMeta
from unstable.utils.logging import setup_logger
from unstable.utils.templates import ACTION_EXTRACTION, OBSERVATION_FORMATTING



class CallableActorWrapper:
    def __init__(self, actor: VLLMActor, lora_path: str|Path, obs_fmt_fn: Callable[[str],str], extract_fn: Callable[[str], Tuple[str, Dict[str, Any]]]) -> None:
        self._actor, self._lora, self._fmt, self._extract = actor, lora_path, obs_fmt_fn, extract_fn

    def __call__(self, observation: str) -> str: 
        _, extracted, _, _, _ = self.act_full(observation)
        return extracted

    def act_full(self, observation: str) -> Tuple[str, str, str, dict, float]:
        prompt = self._fmt(observation=observation)
        try:
            raw, cum_logp, gen_tok = ray.get(self._actor.submit_prompt.remote(prompt=prompt, lora_path=self._lora))
        except Exception as e:
            logging.getLogger("collector").exception(f"act_full submit_prompt failed: {e}")
            raise
        # average per-token logp for generated tokens; guard against zero tokens
        avg_logp = (cum_logp / gen_tok) if gen_tok and gen_tok > 0 else float(cum_logp)
        logging.getLogger("collector").debug(f"Generated tokens: {gen_tok}, cum_logp: {cum_logp:.4f}, avg_logp: {avg_logp:.4f}")
        extracted, format_feedback = self._extract(raw_action=raw)
        return raw, extracted, prompt, format_feedback, avg_logp

@ray.remote(num_cpus=0)
def run_game(game_spec: GameSpec, actor: VLLMActor):
    game_information = GameInformation(game_idx=game_spec.game_idx, eval_model_pid=game_spec.eval_model_pid, eval_opponent_name=game_spec.eval_opponent_name)
    agents = {agent_spec.pid: {
        "traj": PlayerTrajectory(pid=agent_spec.pid) if agent_spec.collect_data else None, 
        "name": agent_spec.lora_path if agent_spec.lora_path else agent_spec.openrouter_name,
        "model": CallableActorWrapper(actor=actor, lora_path=agent_spec.lora_path, obs_fmt_fn=OBSERVATION_FORMATTING[agent_spec.prompt_template], extract_fn=ACTION_EXTRACTION[agent_spec.action_extraction_fn]) if agent_spec.openrouter_name==None else ta.agents.OpenRouterAgent(agent_spec.openrouter_name)
    } for agent_spec in game_spec.agent_specs} # build agents
    env=ta.make(game_spec.env_id); env.reset(num_players=len(agents), seed=game_spec.seed); env.state.error_allowance=0; turn=0
    while True:
        pid, obs = env.get_observation()
        # get model (or opponent) action
        if agents[pid]["traj"] == None:
            raw = extracted = agents[pid]["model"](obs) # fix opponent
            logging.getLogger("collector").debug(f"Opponent pid={pid} produced action.")
        else:
            try:
                raw, extracted, prompt, format_feedback, logp = agents[pid]["model"].act_full(obs)
            except Exception as e:
                logging.getLogger("collector").exception(f"act_full failed for pid={pid}: {e}")
                raise
        done, step_info = env.step(extracted); turn+= 1 # execute the action & increment turn counter
        # general tracking
        game_information.pid.append(pid); game_information.obs.append(obs); game_information.full_actions.append(raw)
        game_information.extracted_actions.append(extracted); game_information.step_infos.append(step_info); game_information.names[pid] = agents[pid]["name"]
        # player specific trackering
        if agents[pid]["traj"] != None:
            agents[pid]["traj"].obs.append(obs); agents[pid]["traj"].actions.append(raw); agents[pid]["traj"].extracted_actions.append(extracted); agents[pid]["traj"].logps.append(logp)
            format_feedback["invalid_move"] = False; agents[pid]["traj"].format_feedbacks.append(format_feedback); agents[pid]["traj"].step_infos.append(step_info)
            if turn % 10 == 0:
                ol = len(agents[pid]["traj"].obs); al = len(agents[pid]["traj"].actions); ll = len(agents[pid]["traj"].logps)
                logging.getLogger("collector").info(f"turn={turn} pid={pid} lengths obs={ol} acts={al} logps={ll}")
        if done: break
    final_rewards, game_info = env.close()
    for pid in agents.keys():
        if agents[pid]["traj"]!=None:
            agents[pid]["traj"].final_reward=final_rewards[pid]; agents[pid]["traj"].game_info=game_info[pid]; agents[pid]["traj"].num_turns=turn
            ol = len(agents[pid]["traj"].obs); al = len(agents[pid]["traj"].actions); ll = len(agents[pid]["traj"].logps)
            if not (ol == al == ll):
                logging.getLogger("collector").warning(f"END GAME length mismatch pid={pid}: obs={ol} acts={al} logps={ll}")
    
    if game_info[pid]["invalid_move"] and agents[pid]["traj"]!=None: agents[pid]["traj"].format_feedbacks[-1]["invalid_move"]=True
    game_information.final_rewards=final_rewards; game_information.num_turns=turn; game_information.game_info=game_info
    return game_information, [agents[pid]["traj"] for pid in agents.keys() if agents[pid]["traj"]!=None]


@ray.remote
class TaskMeta:
    def __init__(self, type: str, env_id: str, actor_id: int = -1):
        self.type = type
        self.env_id = env_id
        self.actor_id = actor_id  # 跟踪使用的actor ID

@ray.remote
class Collector:
    def __init__(self, vllm_config, tracker, buffer, game_scheduler):
        self.logger = setup_logger("collector", ray.get(tracker.get_log_dir.remote()))
        self.tracker, self.buffer, self.game_scheduler = tracker, buffer, game_scheduler
        
        # 创建actors并初始化可用actor池
        num_gpus = int(ray.available_resources().get("GPU", 0))
        self.actors = [VLLMActor.options(num_gpus=1).remote(cfg=vllm_config, tracker=tracker, name=f"Actor-{i}") for i in range(num_gpus)]
        self.available_actors = list(range(len(self.actors)))  # 使用索引跟踪可用actor
        self.logger.info(f"[DEBUG] Initialized {len(self.actors)} actors, all available")
        
        # 任务跟踪
        self.flight: Dict[ray.ObjectRef, TaskMeta] = {}
        self._num_running = lambda typ: sum(meta.type == typ for meta in self.flight.values())
        self.logger.info("Collector initialized")
    
    def _launch_jobs(self, max_train: int, max_eval: Optional[int]):
        train_running = self._num_running("train")
        eval_running = self._num_running("eval")
        self.logger.info(f"[DEBUG] _launch_jobs called - max_train={max_train}, max_eval={max_eval}, currently running: train={train_running}, eval={eval_running}")
        
        # 检查可用actor数量
        available_actors = len(self.available_actors)
        self.logger.info(f"[DEBUG] Available actors: {available_actors}, total actors: {len(self.actors)}")
        
        while self._num_running("train") < max_train and available_actors > 0: # submit new train game
            try:
                self.logger.info(f"[DEBUG] Requesting next train job from scheduler")
                game_spec: GameSpec = ray.get(self.game_scheduler.next_train_job.remote()) # sample game spec
                self.logger.info(f"[DEBUG] Received train game_spec: {game_spec}")
                
                # 从可用actor池中获取actor
                if not self.available_actors:
                    self.logger.warning(f"[DEBUG] No available actors, waiting for actors to be released")
                    break
                    
                actor_id = self.available_actors.pop(0)
                actor = self.actors[actor_id]
                self.logger.info(f"[DEBUG] Launching train job with actor {actor} (ID: {actor_id})")
                
                ref = run_game.remote(game_spec, actor)
                self.flight[ref] = TaskMeta("train", game_spec.env_id, actor_id)
                available_actors -= 1
                self.logger.info(f"[DEBUG] Added train job to flight, now running: {self._num_running('train')}/{max_train}, available actors: {available_actors}")
            except Exception as exc:
                self.logger.error(f"[DEBUG] Exception in train game: {exc}", exc_info=True)

        while max_eval!=None and self._num_running("eval") < max_eval:
            try:
                self.logger.info(f"[DEBUG] Requesting next eval job from scheduler")
                game_spec: GameSpec = ray.get(self.game_scheduler.next_eval_job.remote())
                self.logger.info(f"[DEBUG] Received eval game_spec: {game_spec}")
                # 处理评估任务
                available_actors = len(self.available_actors)
                while max_eval!=None and self._num_running("eval") < max_eval and available_actors > 0:
                    try:
                        game_spec: GameSpec = ray.get(self.game_scheduler.next_eval_job.remote())
                        
                        # 从可用actor池中获取actor
                        if not self.available_actors:
                            self.logger.warning(f"[DEBUG] No available actors for eval, waiting for actors to be released")
                            break
                            
                        actor_id = self.available_actors.pop(0)
                        actor = self.actors[actor_id]
                        self.logger.info(f"[DEBUG] Launching eval job with actor {actor} (ID: {actor_id})")
                        
                        ref = run_game.remote(game_spec, actor)
                        self.flight[ref] = TaskMeta("eval", game_spec.env_id, actor_id)
                        available_actors -= 1
                    except Exception as exc:
                        self.logger.error(f"[DEBUG] Exception in eval game: {exc}", exc_info=True)
            except Exception as exc:
                self.logger.error(f"[DEBUG] Exception in eval game: {exc}", exc_info=True)

    def _handle_finished_job(self, ref):
        meta = self.flight.pop(ref)
        self.logger.info(f"[DEBUG] _handle_finished_job called for {meta.type} job, env={meta.env_id}, actor_id={meta.actor_id}")
        
        # 将actor放回可用池
        if meta.actor_id >= 0 and meta.actor_id < len(self.actors):
            self.available_actors.append(meta.actor_id)
            self.logger.info(f"[DEBUG] Returned actor {meta.actor_id} to available pool, now {len(self.available_actors)} available")
        
        try: 
            self.logger.info(f"[DEBUG] Getting results for {meta.type} job")
            game_information, player_trajs = ray.get(ref)
            self.logger.info(f"[DEBUG] Successfully got results for {meta.type} job")
        except (RayTaskError, RayActorError) as err: 
            self.logger.error(f"[DEBUG] Remote episode failed for {meta.type} task: env={meta.env_id}: {err}", exc_info=True)
            return
        
        if meta.type == "train":
            self.logger.info(f"[DEBUG] Processing train job results")
            self._post_train(meta, game_information, player_trajs)
        else:
            self.logger.info(f"[DEBUG] Processing eval job results")
            self._post_eval(meta, game_information)
    
    def _post_train(self, meta: TaskMeta, game_information: GameInformation, player_trajs: List[PlayerTrajectory]):
        self.logger.info(f"[DEBUG] _post_train called with {len(player_trajs)} trajectories for env={meta.env_id}")
        for i, traj in enumerate(player_trajs):
            self.logger.info(f"[DEBUG] Adding trajectory {i+1}/{len(player_trajs)} to buffer and tracker")
            self.buffer.add_player_trajectory.remote(traj, env_id=meta.env_id)
            self.tracker.add_player_trajectory.remote(traj, env_id=meta.env_id)
        self.logger.info(f"[DEBUG] Updating game scheduler with game information")
        self.game_scheduler.update.remote(game_info=game_information)

    def _post_eval(self, meta: TaskMeta, game_information: GameInformation):
        self.tracker.add_eval_game_information.remote(game_information=game_information, env_id=meta.env_id)
    
    def collect(self, num_train_workers: int, num_eval_workers: Optional[int]=None):
        self.logger.info("[DEBUG] Collector.collect started")
        iteration = 0
        import time
        
        while True:  # Run indefinitely, checking buffer state in the loop
            iteration += 1
            self.logger.info(f"[DEBUG] Collector loop iteration {iteration} - checking buffer.continue_collection")
            buffer_continue = ray.get(self.buffer.continue_collection.remote())
            
            if buffer_continue:
                self.logger.info(f"[DEBUG] Buffer continue_collection returned True - launching jobs")
                self._launch_jobs(num_train_workers, num_eval_workers)
                
                if not self.flight: 
                    # No jobs in flight, wait a bit before checking again
                    self.logger.info(f"[DEBUG] No jobs in flight, waiting before checking again")
                    time.sleep(0.5)
                    continue
                    
                self.logger.info(f"[DEBUG] Waiting for jobs to complete - {len(self.flight)} jobs in flight")
                done_ref, remaining_refs = ray.wait(list(self.flight), num_returns=1)
                self.logger.info(f"[DEBUG] Job completed, handling finished job. {len(remaining_refs)} jobs still in flight")
                self._handle_finished_job(done_ref[0])
            else:
                # Buffer not collecting, wait before checking again
                self.logger.info(f"[DEBUG] Buffer continue_collection returned False - waiting before checking again")
                time.sleep(2.0)  # Wait longer when buffer is not collecting
                
                # Print detailed status
                self.logger.info(f"[DEBUG] Collector status: iteration={iteration}, flight_size={len(self.flight)}, train_running={self._num_running('train')}, eval_running={self._num_running('eval')}")
                
                # Try to get buffer size for debugging
                try:
                    buffer_size = ray.get(self.buffer.size.remote())
                    self.logger.info(f"[DEBUG] Current buffer size: {buffer_size}")
                except Exception as exc:
                    self.logger.error(f"[DEBUG] Failed to get buffer size: {exc}")

