import ray, time
from typing import List, Sequence, Optional

import unstable


def _default_vllm_cfg(model_name: str, lora_cfg: dict) -> dict: return {"model_name": model_name, "temperature": 0.6, "max_tokens": 4096, "max_parallel_seq": 128, "max_loras": 8, "lora_config": lora_cfg, "max_model_len": 8192}
def _build_model_sampler(sampler_name: str, model_registry): pass
def _build_env_sampler(sampler_name: str, train_envs: List[unstable.TrainEnvSpec], eval_envs: List[unstable.EvalEnvSpec]): pass
class _UBRun:
    def __init__(self, *, collector, learner, tracker, model_registry, step_buffer, game_scheduler, iterations: int, num_workers: int, num_eval_workers: Optional[int]) -> None:
        self.collector, self.learner, self.tracker, self.model_registry, self.step_buffer, self.game_scheduler = collector, learner, tracker, model_registry, step_buffer, game_scheduler
        self._iterations, self._num_workers, self._num_eval = iterations, num_workers, num_eval_workers
        self._collector_ref, self._learner_ref = None, None

    def start(self) -> None:
        if self._collector_ref is None: self._collector_ref = self.collector.collect.remote(num_train_workers=self._num_workers, num_eval_workers=self._num_eval)
        if self._learner_ref is None:   self._learner_ref = self.learner.train.remote(self._iterations)

    def wait(self, poll_seconds: float = 5.0) -> None:
        if self._learner_ref is None: raise RuntimeError("start() must be called before wait().")
        while True:
            done, _ = ray.wait([self._learner_ref], timeout=poll_seconds)
            if done: break
            time.sleep(poll_seconds)

    def stop(self, kill_collector: bool = True, shutdown_ray: bool = False) -> None:
        if kill_collector and self._collector_ref is not None:  ray.kill(self.collector, no_restart=True)
        if shutdown_ray:                                        ray.shutdown()


def build(
    *, model_name: str, train_envs: Sequence[unstable.TrainEnvSpec], eval_envs: Optional[Sequence[unstable.EvalEnvSpec]]=None, algorithm: str="reinforce", buffer_size: Optional[int]=None, batch_size: int=384, mini_batch_size: int=1, learning_rate: float=1e-5, grad_clip: float=0.2, iterations: int=200, 
    lora_cfg: Optional[dict]=None, vllm_cfg: Optional[dict]=None, activation_checkpointing: bool=True, gradient_checkpointing: bool=True, use_trainer_cache: bool=False, opponent_fixed: Sequence[str]=(), opponent_strategy: str="mirror", num_train_workers: int=256, num_eval_workers: int|None=16,
) -> "_UBRun":
# TODO add default lora cfg
    ray.init(namespace="unstable", ignore_reinit_error=True) # build ray
    run_name = f"{prefix}-{int(time.time())}"
    tracker = Tracker.options(name="Tracker").remote(run_name=run_name) # build tracker
    registry = ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker) # build model registry
    registry.add_checkpoint.remote(uid="base", path=None, iteration=0) # add base model to registry
    for name in opponent_fixed: registry.add_fixed.remote(name) # add fixed models to registry
    step_buffer = (unstable.EpisodeBuffer if algorithm=="a2c" else unstable.StepBuffer).options(name="Buffer").remote(max_buffer_size=buffer_size or batch_size*2, tracker=tracker, final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]), step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]), sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]))
    game_scheduler = GameScheduler.options(name="GameScheduler").remote(model_sampler=_build_model_sampler(opponent_strategy, registry), env_sampler=_build_env_sampler(list(train_envs), list(eval_envs))) # build samplers and game scheduler
    collector = Collector.options(name="Collector").remote(vllm_config=vllm_cfg or _default_vllm_cfg(model_name, lora_cfg or {}), tracker=tracker, buffer=step_buffer, game_scheduler=game_scheduler) # build collector
    learner = (REINFORCELearner if algorithm.lower() == "reinforce" else A2CLearner).options(num_gpus=learners, name="Learner").remote(model_name=model_name, lora_cfg=lora_cfg, batch_size=batch_size, mini_batch_size=mini_batch_size, learning_rate=learning_rate, grad_clip=grad_clip, buffer=step_buffer, tracker=tracker, model_registry=registry, activation_checkpointing=activation_checkpointing, gradient_checkpointing=gradient_checkpointing, use_trainer_cache=use_trainer_cache)
    if algorithm.lower() == "reinforce":    ray.get(learner.initialize_algorithm.remote(max_train_len=None, max_generation_len=vllm_cfg["max_tokens"]))
    elif algorithm.lower() == "a2c":        ray.get(learner.initialize_algorithm.remote(infer_mini_batch_size=16, critic_learning_rate=learning_rate, normalize_adv=True))
    return _UBRun(collector=collector, learner=learner, tracker=tracker, model_registry=registry, step_buffer=step_buffer, game_scheduler=game_scheduler, iterations=iterations, num_workers=num_train_workers, num_eval_workers=num_eval_workers)
