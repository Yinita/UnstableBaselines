from .registry.model_registry import ModelRegistry
from .samplers.model import build_sampler   # factory returns the right class


class _UBRun:
    def __init__(self, *, collector, learner, tracker, model_sampler, step_buffer, scheduler=None, iterations: int, num_workers: int, num_eval_workers: int):
        self.collector = collector
        self.learner = learner
        self.tracker = tracker
        self.model_sampler = model_sampler
        self.step_buffer = step_buffer
        self.scheduler = scheduler

        self._iterations = iterations
        self._num_workers = num_workers
        self._num_eval = num_eval_workers

        self._collector_ref = None
        self._learner_ref   = None

    def start(self) -> None:
        if self._collector_ref is None: self._collector_ref = self.collector.collect.remote(num_workers=self._num_workers, num_eval_workers=self._num_eval)
        if self._learner_ref is None:   self._learner_ref = self.learner.train.remote(self._iterations)

    def wait(self, poll_seconds: float = 5.0) -> None:
        if self._learner_ref is None: raise RuntimeError("start() must be called before wait().")
        while True:
            done, _ = ray.wait([self._learner_ref], timeout=poll_seconds)
            if done: break
            time.sleep(poll_seconds)

    def stop(self, kill_collector: bool = True, shutdown_ray: bool = False) -> None:
        if kill_collector and self._collector_ref is not None:  ray.kill(self.collector, no_restart=True) # kill the actor so GPU workers shut down promptly
        if shutdown_ray:                                        ray.shutdown()

def build(model_name: str, train_envs, eval_envs=None, opponent_fixed=(), opponent_strategy="mirror", learners=1, actors="all-remaining", buffer_size=None, batch_size=384, iterations=200, **kwargs):
    ray.init(namespace="unstable", ignore_reinit_error=True)

    tracker = Tracker.options(name="Tracker").remote(run_name=_make_run_name())
    registry = ModelRegistry.options(name="ModelRegistry").remote()
    registry.add_checkpoint.remote("ckpt-0", None, None)
    for name in opponent_fixed: registry.add_fixed.remote(name)
    step_buffer = StepBuffer.options(name="StepBuffer").remote(
        max_buffer_size = buffer_size or batch_size*2,
        tracker = tracker,
        final_reward_transformation = retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
        step_reward_transformation  = retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
        sampling_reward_transformation = retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
    )
    sampler = build_sampler(opponent_strategy, registry)
    env_sched = EnvScheduler.options(name="EnvScheduler").remote(train_env_specs=train_envs, eval_env_specs=eval_envs or train_envs, opponent_sampler_state=ray.put(sampler))
    collector = Collector.options(name="Collector").remote(tracker=tracker, vllm_config=_default_vllm_cfg(model_name), step_buffer=step_buffer, env_sched=env_sched)
    learner = StandardLearner.options(num_gpus=learners).remote(tracker=tracker, model_name=model_name, step_buffer=step_buffer, registry=registry, batch_size=batch_size, iterations=iterations, **kwargs)
    return _UBRun(tracker, registry, collector, learner, step_buffer)
