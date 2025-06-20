import re, random, logging, itertools
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Protocol

import ray
import textarena as ta
from ray.exceptions import RayActorError, RayTaskError

# local imports
from unstable.actor import VLLMActor
from unstable.core import BaseTracker, Trajectory, EpisodeResult, PlaySpec, TaskMeta
from unstable.utils.logging import setup_logger
from unstable.utils.templates import ACTION_EXTRACTION, OBSERVATION_FORMATTING

assert ta.__version__ == "0.6.9", f"TextArena 0.6.9 required, currently {ta.__version__}"


class CallableActorWrapper:
    """ Thin synchronous wrapper around a remote ``VLLMActor`` that can return either just the extracted action or the *full* reasoning string + formatted prompt """
    def __init__(self, actor: VLLMActor, lora_path: str|Path, obs_fmt_fn: Callable[[str],str], extract_fn: Callable[[str], Tuple[str, Dict[str, Any]]]) -> None:
        self._actor = actor
        self._lora = str(lora_path)
        self._fmt = obs_fmt_fn
        self._extract = extract_fn

    def __call__(self, observation: str) -> str:
        raw, extracted, _ = self.act_full(observation)
        return extracted

    def act_full(self, observation: str) -> Tuple[str, str, str]:
        """Return *(raw_response, extracted_action, formatted_prompt)*."""
        prompt = self._fmt(observation=observation)
        raw = ray.get(self._actor.submit_prompt.remote(prompt=prompt, lora_path=self._lora))
        extracted, _ = self._extract(raw_action=raw)
        return raw, extracted, prompt


def _iter_from_uid(uid: str) -> int:
    """Return integer suffix from “ckpt‑42”; 0 if no match."""
    return int(m.group(1)) if (m := re.search(r"(\d+)$", uid)) else 0


def _extract_action(action: str) -> str:
    """Return **lower‑cased** content inside the first ``[...]`` pair."""
    return (m.group(1).strip().lower() if (m := re.search(r"\[(.*?)\]", action)) else "")


@ray.remote(num_cpus=0)
def play_episode(spec: Dict[str, Any]) -> EpisodeResult:  # noqa: C901
    spec = PlaySpec(**spec)
    env = ta.make(spec.env_id); env.reset(num_players=spec.num_players,seed=spec.seed); env.state.error_allowance=0
    traj = Trajectory(); turn = 0
    action_seq: List[str] = []
    while True:
        pid, obs = env.get_observation()
        if pid == spec.player_id and hasattr(spec.agents[pid], "act_full"): 
            raw, extracted, prompt = spec.agents[pid].act_full(obs)
        else:
            extracted = spec.agents[pid](obs)
        done, info = env.step(extracted)
        if pid == spec.player_id: # Only track the learner’s internal details.
            traj.pid.append(pid); traj.obs.append(prompt); traj.actions.append(raw)
            traj.extracted_actions.append(extracted); traj.infos.append(info)
        if done: break
        action_seq.append(_extract_action(extracted))
        turn += 1
    traj.final_rewards = env.close()
    traj.num_turns = turn
    end_by_opp_inv = info["end_by_invalid"] and pid != spec.player_id
    return EpisodeResult(traj=traj, end_by_opponent_invalid=end_by_opp_inv, action_seq=action_seq, final_rewards=traj.final_rewards)

@ray.remote
class Collector:
    """Main orchestrator for training & evaluation rollouts."""
    def __init__(
        self,
        num_actors: int,
        step_buffer,
        model_pool,
        tracker: BaseTracker,
        vllm_config: Dict[str, Any],
        training_envs: List[tuple[str, int, str|None]],
        evaluation_envs: List[tuple[str, int, str|None]],
        evaluation_opponent: str = "google/gemini-2.0-flash-lite-001",
        max_eval_games_per_ckpt: int = 32,
        filter_opponent_invalid: bool = False,
        action_extraction: str = "default",
    ) -> None:
        self.logger = setup_logger("collector", ray.get(tracker.get_log_dir.remote()))
        self.buffer = step_buffer
        self.model_pool = model_pool
        self.tracker = tracker
        self.train_envs = training_envs
        self.eval_envs = evaluation_envs
        self.eval_opponent = evaluation_opponent
        self.max_eval = max_eval_games_per_ckpt
        self.filter_invalid = filter_opponent_invalid
        self.extract_key = action_extraction
        self.actors = [VLLMActor.options(num_gpus=1).remote(vllm_config=vllm_config, tracker=tracker, name=f"Actor-{i}") for i in range(num_actors)]
        self._actor_iter = itertools.cycle(self.actors)
        self.rng_train = random.Random(489)
        self.rng_eval = random.Random(977)
        self.flight: Dict[ray.ObjectRef, TaskMeta] = {}
        self.eval_counter: Dict[str, int] = {}

    def _next_actor(self):                  return next(self._actor_iter)
    def _obs_extract(self, tmpl):           return OBSERVATION_FORMATTING[tmpl], ACTION_EXTRACTION[self.extract_key]
    def _num_running(self, typ: str) -> int:return sum(meta.type == typ for meta in self.flight.values())

    def _sample_env(self, rng: random.Random, envs):
        env_id, n, tmpl = rng.choice(envs)
        pid = rng.randrange(n)
        return env_id, n, pid, tmpl

    def _make_learner(self, lora_path, tmpl):
        obs_fmt, ext = self._obs_extract(tmpl)
        return CallableActorWrapper(self._next_actor(), lora_path, obs_fmt, ext)

    def _make_opponent(self, opp_path, tmpl):
        if opp_path.startswith("ckpt-"):
            obs_fmt, ext = self._obs_extract(tmpl)
            return CallableActorWrapper(self._next_actor(), opp_path, obs_fmt, ext)
        return ta.agents.OpenRouterAgent(opp_path)

    def collect(self, num_workers: int, num_eval_workers: int):
        while ray.get(self.buffer.continue_collection.remote()):
            self._launch_jobs(num_workers, num_eval_workers)
            if not self.flight: continue
            done_ref, _ = ray.wait(list(self.flight), num_returns=1)
            self._handle_finished(done_ref[0])

    def _launch_jobs(self, max_train: int, max_eval: int):
        while self._num_running("train") < max_train:
            self._submit_train()

        latest_ckpt = ray.get(self.model_pool.latest_ckpt.remote())
        if self.eval_counter.get(latest_ckpt, 0) < self.max_eval * len(self.eval_envs):
            while self._num_running("eval") < max_eval:
                self._submit_eval(latest_ckpt)

    def _submit_train(self):
        env_id, n, pid, tmpl = self._sample_env(self.rng_train, self.train_envs)
        current_uid = ray.get(self.model_pool.latest_ckpt.remote())
        lora_path = ray.get(self.model_pool.ckpt_path.remote(current_uid))
        opp_uid = ray.get(self.model_pool.sample.remote(uid_me=current_uid))
        opp_path = ray.get(self.model_pool.ckpt_path.remote(opp_uid))

        learner = self._make_learner(lora_path, tmpl)
        opponent = self._make_opponent(opp_path, tmpl)

        spec = PlaySpec(env_id, n, pid, [learner if i == pid else opponent for i in range(n)], self.rng_train.getrandbits(32))
        ref = play_episode.remote(asdict(spec))
        self.flight[ref] = TaskMeta("train", env_id, pid, spec.seed, current_uid, opp_uid)
        self.logger.debug(f"↪ train {_iter_from_uid(current_uid)} {env_id} pid={pid}")

    def _submit_eval(self, ckpt_uid):
        env_id, n, pid, tmpl = self._sample_env(self.rng_eval, self.eval_envs)
        lora_path = ray.get(self.model_pool.ckpt_path.remote(ckpt_uid))
        learner = self._make_learner(lora_path, tmpl)
        opponent = self._make_opponent(self.eval_opponent, tmpl)
        spec = PlaySpec(env_id, n, pid, [learner if i == pid else opponent for i in range(n)], self.rng_eval.getrandbits(32))
        ref = play_episode.remote(asdict(spec))
        self.flight[ref] = TaskMeta("eval", env_id, pid, spec.seed, ckpt_uid)
        self.eval_counter[ckpt_uid] = self.eval_counter.get(ckpt_uid, 0) + 1
        self.logger.debug(f"↪ eval {_iter_from_uid(ckpt_uid)} {env_id} pid={pid}")

    def _handle_finished(self, ref):
        meta = self.flight.pop(ref)
        try:                                            res: EpisodeResult = ray.get(ref)
        except (RayTaskError, RayActorError) as err:    self.logger.exception("remote episode failed", exc_info=err); return
        match meta.type:
            case "train":   self._post_train(meta, res)
            case "eval":    self._post_eval(meta, res)
            case _:         self.logger.warning(f"unknown task type {meta.type}") # pragma: no cover

    def _post_train(self, meta: TaskMeta, res: EpisodeResult):
        self.logger.info("train_done", extra=dict(env=meta.env_id, ckpt=meta.ckpt_uid, length=len(res.traj.pid), invalid=res.end_by_opponent_invalid))
        if self.filter_invalid and res.end_by_opponent_invalid: return
        self.buffer.add_trajectory.remote(res.traj, meta.player_id, meta.env_id)
        self.tracker.add_trajectory.remote(res.traj, meta.player_id, meta.env_id)

        if meta.opponent_uid:
            self.model_pool.push_game_outcome.remote(
                uid_me=meta.ckpt_uid, uid_opp=meta.opponent_uid,
                final_reward=res.traj.final_rewards[meta.player_id], game_action_seq=res.action_seq,
            )

    def _post_eval(self, meta: TaskMeta, res: EpisodeResult):
        if self.tracker:
            self.tracker.add_eval_episode.remote(
                episode_info=None, final_rewards=res.final_rewards, player_id=meta.player_id,
                env_id=meta.env_id, iteration=meta.ckpt_uid,
            )
        self.logger.info("eval_done", extra=dict(env=meta.env_id, ckpt=meta.ckpt_uid, seed=meta.seed, reward=res.final_rewards[meta.player_id]))