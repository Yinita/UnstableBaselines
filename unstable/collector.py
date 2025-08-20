import re, random, logging, itertools, time, os
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
from unstable.utils import write_samples_to_file
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
            # Handle both cases: when submit_prompt returns a single value or a tuple of three values
            result = ray.get(self._actor.submit_prompt.remote(prompt=prompt, lora_path=self._lora))
            
            # Check if result is a tuple with at least 3 elements
            if isinstance(result, tuple) and len(result) >= 3:
                raw, cum_logp, gen_tok = result
            else:
                # If it's a single value (string), use default values for cum_logp and gen_tok
                raw = result
                cum_logp = 0.0
                gen_tok = len(raw.split())  # Approximate token count
                logging.getLogger("collector").info(f"submit_prompt returned a single value, using defaults: cum_logp={cum_logp}, gen_tok={gen_tok}")
        except Exception as e:
            logging.getLogger("collector").exception(f"act_full submit_prompt failed: {e}")
            raise
        # average per-token logp for generated tokens; guard against zero tokens
        avg_logp = (cum_logp / gen_tok) if gen_tok and gen_tok > 0 else 0.0
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
    try:
        cd_map = {pid: (agents[pid]["traj"] is not None) for pid in agents}
        logging.getLogger("collector").info(f"game_idx={game_spec.game_idx} collect_data map: {cd_map}")
    except Exception:
        pass
    env=ta.make(game_spec.env_id); env.reset(num_players=len(agents), seed=game_spec.seed); env.state.error_allowance=0; turn=0
    while True:
        pid, obs = env.get_observation()
        # get model (or opponent) action
        if agents[pid]["traj"] is None:
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
        if agents[pid]["traj"] is not None:
            agents[pid]["traj"].obs.append(obs); agents[pid]["traj"].actions.append(raw); agents[pid]["traj"].extracted_actions.append(extracted); agents[pid]["traj"].logps.append(logp)
            format_feedback["invalid_move"] = False; agents[pid]["traj"].format_feedbacks.append(format_feedback); agents[pid]["traj"].step_infos.append(step_info)
            if turn % 10 == 0:
                ol = len(agents[pid]["traj"].obs); al = len(agents[pid]["traj"].actions); ll = len(agents[pid]["traj"].logps)
                logging.getLogger("collector").info(f"turn={turn} pid={pid} lengths obs={ol} acts={al} logps={ll}")
        if done: break
    final_rewards, game_info = env.close()
    for pid in agents.keys():
        if agents[pid]["traj"] is not None: 
            agents[pid]["traj"].final_reward=final_rewards[pid]
            agents[pid]["traj"].game_info=game_info[pid]
            agents[pid]["traj"].num_turns=turn
        if game_info[pid]["invalid_move"] and agents[pid]["traj"] is not None: 
            agents[pid]["traj"].format_feedbacks[-1]["invalid_move"]=True
    game_information.final_rewards=final_rewards; game_information.num_turns=turn; game_information.game_info=game_info
    try:
        lengths = {pid: {
            "obs": len(agents[pid]["traj"].obs) if agents[pid]["traj"] is not None else 0,
            "acts": len(agents[pid]["traj"].actions) if agents[pid]["traj"] is not None else 0,
            "extracted": len(agents[pid]["traj"].extracted_actions) if agents[pid]["traj"] is not None else 0,
            "logps": len(agents[pid]["traj"].logps) if agents[pid]["traj"] is not None else 0,
        } for pid in agents}
        logging.getLogger("collector").info(f"game_idx={game_spec.game_idx} final traj lengths per pid: {lengths}; turns={turn}")
    except Exception:
        pass
    return game_information, [agents[pid]["traj"] for pid in agents.keys() if agents[pid]["traj"]!=None]


@ray.remote
class Collector:
    def __init__(self, vllm_config, tracker, buffer, game_scheduler):
        self.logger = setup_logger("collector", ray.get(tracker.get_log_dir.remote()))
        self.tracker, self.buffer, self.game_scheduler = tracker, buffer, game_scheduler
        num_gpus = int(ray.available_resources().get("GPU", 0))
        if num_gpus == 0:
            raise RuntimeError("No GPUs available for VLLMActor initialization")
        self.actors = [VLLMActor.options(num_gpus=1).remote(cfg=vllm_config, tracker=tracker, name=f"Actor-{i}") for i in range(int(num_gpus)-1)]
        self._actor_iter = itertools.cycle(self.actors)

        # thead keeping
        self.flight: Dict[ray.ObjectRef, TaskMeta] = {}
        self._num_running = lambda typ: sum(meta.type == typ for meta in self.flight.values())
        self.logger.info("Collector initialized")

        # Directory to save collected samples (dynamic under current run log dir)
        run_log_dir = ray.get(tracker.get_log_dir.remote())
        self.samples_dir = os.path.join(run_log_dir, "samples")
        try:
            os.makedirs(self.samples_dir, exist_ok=True)
            self.logger.info(f"Samples directory ensured at: {self.samples_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create samples directory: {e}")
    
    def _launch_jobs(self, max_train: int, max_eval: Optional[int]):
        # 训练任务调度
        while self._num_running("train") < max_train: # submit new train game
            try:
                # 获取训练游戏规格
                self.logger.info("Requesting next training job from scheduler...")
                game_spec_ref = self.game_scheduler.next_train_job.remote()
                game_spec: GameSpec = ray.get(game_spec_ref) # sample game spec
                
                # 验证游戏规格
                if game_spec is None:
                    self.logger.error("Received None game_spec from next_train_job")
                    # 短暂休眠避免过于频繁的请求
                    time.sleep(1)
                    continue
                    
                self.logger.info(f"Received train game_spec: game_idx={game_spec.game_idx}, "
                              f"env_id={game_spec.env_id}, seed={game_spec.seed}, "
                              f"num_agents={len(game_spec.agent_specs)}")
                
                # 获取执行器并提交任务
                actor: VLLMActor = next(self._actor_iter) # get actor
                ref = run_game.remote(game_spec, actor)
                self.flight[ref] = TaskMeta("train", game_spec.env_id)
                self.logger.info(f"Submitted train job: game_idx={game_spec.game_idx}")
            except Exception as exc:
                self.logger.error(f"Exception in train job scheduling: {exc}", exc_info=True)
                # 短暂休眠避免过于频繁的请求
                time.sleep(1)

        # 评估任务调度
        while max_eval!=None and self._num_running("eval") < max_eval:
            try:
                # 获取评估游戏规格
                self.logger.info("Requesting next evaluation job from scheduler...")
                game_spec_ref = self.game_scheduler.next_eval_job.remote()
                game_spec: GameSpec = ray.get(game_spec_ref)
                
                # 验证游戏规格
                if game_spec is None:
                    self.logger.error("Received None game_spec from next_eval_job")
                    # 短暂休眠避免过于频繁的请求
                    time.sleep(1)
                    continue
                    
                self.logger.info(f"Received eval game_spec: game_idx={game_spec.game_idx}, "
                              f"env_id={game_spec.env_id}, seed={game_spec.seed}, "
                              f"eval_model_pid={game_spec.eval_model_pid}, "
                              f"eval_opponent_name={game_spec.eval_opponent_name}")
                
                # 获取执行器并提交任务
                actor: VLLMActor = next(self._actor_iter) # get actor
                ref = run_game.remote(game_spec, actor)
                self.flight[ref] = TaskMeta("eval", game_spec.env_id)
                self.logger.info(f"Submitted eval job: game_idx={game_spec.game_idx}")
            except Exception as exc:
                self.logger.error(f"Exception in eval job scheduling: {exc}", exc_info=True)
                # 短暂休眠避免过于频繁的请求
                time.sleep(1)

    def _handle_finished_job(self, ref):
        meta = self.flight.pop(ref)
        try:
            game_information, player_trajs = ray.get(ref)
        except (RayTaskError, RayActorError) as err:
            self.logger.error(f"Remote episode failed for {meta.type} task: env={meta.env_id}: {err}", exc_info=True)
            return
        self.logger.info(f"Finished {meta.type} job: game_idx={game_information.game_idx}, env_id={meta.env_id}")
        if meta.type == "train":
            self._post_train(meta, game_information, player_trajs)
        else:
            self._post_eval(meta, game_information, player_trajs)
    
    def _post_train(self, meta: TaskMeta, game_information: GameInformation, player_trajs: List[PlayerTrajectory]):
        for traj in player_trajs: self.buffer.add_player_trajectory.remote(traj, env_id=meta.env_id); self.tracker.add_player_trajectory.remote(traj, env_id=meta.env_id)
        self.game_scheduler.update.remote(game_info=game_information)
        # Persist collected samples for this game
        try:
            filename = os.path.join(self.samples_dir, f"samples_game_{game_information.game_idx}.csv")
            write_samples_to_file(player_trajs, filename, env_id=meta.env_id)
            # diagnostics
            lengths = [
                {
                    "pid": traj.pid,
                    "obs": len(traj.obs),
                    "acts": len(traj.actions),
                    "extracted": len(traj.extracted_actions),
                    "logps": len(traj.logps),
                } for traj in player_trajs
            ]
            total_rows = sum(min(len(t.obs), max(len(t.actions), len(t.extracted_actions))) for t in player_trajs)
            self.logger.info(f"Saved collected samples to {filename}; total_rows={total_rows}; traj_lengths={lengths}")
            # also persist game information for debugging
            from unstable.utils.misc import write_game_information_to_file
            gi_file = os.path.join(self.samples_dir, f"game_info_{game_information.game_idx}.csv")
            write_game_information_to_file(game_information, gi_file)
            self.logger.info(f"Saved game information to {gi_file}")
        except Exception as e:
            self.logger.error(f"Error saving collected samples: {e}")

    def _post_eval(self, meta: TaskMeta, game_information: GameInformation, player_trajs: List[PlayerTrajectory]):
        # always track eval game info
        self.tracker.add_eval_game_information.remote(game_information=game_information, env_id=meta.env_id)
        # persist eval samples if any trajectories were collected (eval_model_pid may have collect_data=True)
        if player_trajs:
            try:
                filename = os.path.join(self.samples_dir, f"eval_samples_game_{game_information.game_idx}.csv")
                write_samples_to_file(player_trajs, filename, env_id=meta.env_id)
                lengths = [
                    {
                        "pid": traj.pid,
                        "obs": len(traj.obs),
                        "acts": len(traj.actions),
                        "extracted": len(traj.extracted_actions),
                        "logps": len(traj.logps),
                    } for traj in player_trajs
                ]
                total_rows = sum(min(len(t.obs), max(len(t.actions), len(t.extracted_actions))) for t in player_trajs)
                self.logger.info(f"Saved eval samples to {filename}; total_rows={total_rows}; traj_lengths={lengths}")
                from unstable.utils.misc import write_game_information_to_file
                gi_file = os.path.join(self.samples_dir, f"eval_game_info_{game_information.game_idx}.csv")
                write_game_information_to_file(game_information, gi_file)
                self.logger.info(f"Saved eval game information to {gi_file}")
            except Exception as e:
                self.logger.error(f"Error saving eval samples: {e}")
    
    def collect(self, num_train_workers: int, num_eval_workers: Optional[int]=None):
        self.logger.info("entered collect func")
        while ray.get(self.buffer.continue_collection.remote()):
            self.logger.info("entered collect loop")
            self._launch_jobs(num_train_workers, num_eval_workers)
            if not self.flight:
                # nothing inflight yet; short sleep to avoid a tight loop
                time.sleep(0.5)
                continue
            # Wait with timeout to avoid appearing stuck; log heartbeat if nothing finishes
            done_ref, _ = ray.wait(list(self.flight), num_returns=1, timeout=60.0)
            if not done_ref:
                running_train = self._num_running("train")
                running_eval = self._num_running("eval")
                resources = None
                try:
                    resources = ray.available_resources()
                except Exception:
                    resources = {}
                self.logger.info(
                    f"Heartbeat: waiting... flight={len(self.flight)}, running_train={running_train}, running_eval={running_eval}, resources={resources}"
                )
                continue
            self._handle_finished_job(done_ref[0])
