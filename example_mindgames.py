import ray, unstable, time
import unstable.reward_transformations as retra

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_GENERATION_LENGTH = 4096
MAX_TRAIN_SEQ_LEN = None # if you are running out of vRam, you can decrease this.

lora_config = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
vllm_config = {
    "model_name": MODEL_NAME, "temperature": 0.6, "max_tokens": MAX_GENERATION_LENGTH,
    "max_parallel_seq": 128, "max_loras": 8, "lora_config": lora_config,
    "max_model_len": 8192
}

ray.init(namespace="unstable") # the namespace is mostly important for the terminal_interface.py script (which loads the modules from the "unstable" namespace)

# initialize environment scheduler
env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
    train_env_specs=[
        unstable.TrainEnvSpec(
            env_id="Codenames-v0-train", 
            num_players=4, 
            num_actors=4, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.TrainEnvSpec(
            env_id="ColonelBlotto-v0-train", 
            num_players=2, 
            num_actors=2, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.TrainEnvSpec(
            env_id="ThreePlayerIPD-v0-train", 
            num_players=3, 
            num_actors=3, 
            prompt_template="llama-instruct-zs"
        ),
    ],
    eval_env_specs=[
        unstable.EvalEnvSpec(
            env_id="Codenames-v0-train", 
            num_players=4, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.EvalEnvSpec(
            env_id="ColonelBlotto-v0-train", 
            num_players=2, 
            prompt_template="llama-instruct-zs"
        ),
        unstable.EvalEnvSpec(
            env_id="ThreePlayerIPD-v0-train", 
            num_players=3, 
            prompt_template="llama-instruct-zs"
        ),
])


tracker = unstable.Tracker.options(name="Tracker").remote(
    run_name=f"MindGamesTest-{MODEL_NAME.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", 
    wandb_project="UnstableBaselines"
) 

model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
ray.get(model_registry.add_fixed.remote(name="google/gemini-2.0-flash-lite-001"))

model_sampler = unstable.samplers.model_samplers.BaseModelSampler(
    model_registry=model_registry
) 

game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(
    model_sampler=model_sampler, 
    env_sampler=env_sampler, 
    logging_dir=ray.get(tracker.get_log_dir.remote())
)

step_buffer = unstable.StepBuffer.options(name="Buffer").remote(
    max_buffer_size=768, 
    tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([
        retra.RoleAdvantageByEnvFormatter()
    ]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([
        retra.RewardForFormat(1.5), 
        retra.PenaltyForInvalidMove(1.0, -1.0)
    ]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([
        retra.NormalizeRewardsByEnv(True)
    ]),
)

collector = unstable.Collector.options(name="Collector").remote(
    vllm_config=vllm_config, 
    tracker=tracker, 
    buffer=step_buffer, 
    game_scheduler=game_scheduler
)

learner = unstable.REINFORCELearner.options(num_gpus=1, name="Learner").remote(
    model_name=MODEL_NAME,
    lora_cfg=lora_config,
    batch_size=384,
    mini_batch_size=1,
    learning_rate=1e-5,
    grad_clip=0.2,
    buffer=step_buffer,
    tracker=tracker,
    model_registry=model_registry,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False
)
ray.get(learner.initialize_algorithm.remote(
    max_train_len=MAX_TRAIN_SEQ_LEN,
    max_generation_len=MAX_GENERATION_LENGTH
))

try:
    collector.collect.remote(
        num_train_workers=512, 
        num_eval_workers=16
    ) # if you are running out of ram, reduce this
    ray.get(learner.train.remote(200))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()
