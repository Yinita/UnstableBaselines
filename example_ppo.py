import time, ray, unstable
import unstable.reward_transformations as retra
from unstable.learners.ppo_learner import PPOLearner

# --- Hyperparameters ---
COLLECTION_WORKERS = 200
EVALUATION_WORKERS = 16
ITERATIONS = 500
MODEL_NAME = "Qwen/Qwen3-1.7B-Base" # Using a smaller model for broader compatibility
BATCH_SIZE = 64
MINI_BATCH_SIZE = 8
BUFFER_SIZE = 64 * 4
LR = 2e-5
GRAD_CLIP = 1.0
MAX_TRAIN_SEQ_LEN = 3000
MAX_GENERATION_LENGTH = 4096

# PPO Specific Hyperparameters
PPO_CFG = {
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "clip_coef": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "update_epochs": 4,
    "norm_adv": True,
    "max_grad_norm": 0.5,
    "critic_learning_rate": 5e-5
}

lora_config = {
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}

vllm_config = {
    "model_name": MODEL_NAME,
    "temperature": 0.7,
    "max_tokens": MAX_GENERATION_LENGTH,
    "max_parallel_seq": 128,
    "max_loras": 8,
    "lora_config": lora_config,
    "max_model_len": 8192
}

# --- Setup ---
ray.init(namespace="unstable_ppo")

# Environment Sampler with all new games
env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
    train_env_specs=[
        unstable.TrainEnvSpec(env_id="ColonelBlotto-v0-train", num_players=2, num_actors=2, prompt_template="qwen3-zs"),
        unstable.TrainEnvSpec(env_id="ThreePlayerIPD-v0-train", num_players=3, num_actors=3, prompt_template="qwen3-zs"),
        unstable.TrainEnvSpec(env_id="Codenames-v0-train", num_players=4, num_actors=4, prompt_template="qwen3-zs"),
        unstable.TrainEnvSpec(env_id="SecretMafia-v0-train", num_players=6, num_actors=6, prompt_template="qwen3-zs"),
    ],
    eval_env_specs=[
        unstable.EvalEnvSpec(env_id="ColonelBlotto-v0-train", num_players=2, prompt_template="qwen3-zs"),
        unstable.EvalEnvSpec(env_id="ThreePlayerIPD-v0-train", num_players=3, prompt_template="qwen3-zs"),
        unstable.EvalEnvSpec(env_id="Codenames-v0-train", num_players=4, prompt_template="qwen3-zs"),
        unstable.EvalEnvSpec(env_id="SecretMafia-v0-train", num_players=6, prompt_template="qwen3-zs"),
    ])

# Tracker
tracker = unstable.Tracker.options(name="Tracker").remote(
    run_name=f"PPO-{MODEL_NAME.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", 
    wandb_project="UnstableBaselines-PPO"
) 

# Model Registry
model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
# Note: You might need to set OPENROUTER_API_KEY for this to work
ray.get(model_registry.add_fixed.remote(name="google/gemini-2.0-flash-lite-001"))

# Model Sampler
model_sampler = unstable.samplers.model_samplers.BaseModelSampler(model_registry=model_registry) 

# Game Scheduler
game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(model_sampler=model_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()))

# Data Buffer
step_buffer = unstable.EpisodeBuffer.options(name="Buffer").remote(
    max_buffer_size=BUFFER_SIZE,
    tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

# Collector
collector = unstable.Collector.options(name="Collector").remote(
    vllm_config=vllm_config, tracker=tracker, buffer=step_buffer, game_scheduler=game_scheduler,
)

# PPO Learner
learner = PPOLearner.options(num_gpus=1, name="Learner").remote(
    model_name=MODEL_NAME,
    lora_cfg=lora_config,
    batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE,
    learning_rate=LR,
    grad_clip=GRAD_CLIP,
    buffer=step_buffer,
    tracker=tracker,
    model_registry=model_registry,
    activation_checkpointing=True,
    gradient_checkpointing=True,
    use_trainer_cache=False
)
ray.get(learner.initialize_algorithm.remote(
    ppo_cfg=PPO_CFG,
    infer_mini_batch_size=MINI_BATCH_SIZE * 4, # Inference can use a larger batch
    normalize_adv=PPO_CFG.get("norm_adv", True),
    max_train_len=MAX_TRAIN_SEQ_LEN,
    max_generation_len=MAX_GENERATION_LENGTH
))

# --- Run Training ---
print("Starting PPO training run...")
print(f"- Model: {MODEL_NAME}")
print(f"- Games: {env_sampler.env_list()}")
print(f"- Check logs and results in the 'outputs' directory.")

try:
    collector.collect.remote(num_train_workers=COLLECTION_WORKERS, num_eval_workers=EVALUATION_WORKERS)
    ray.get(learner.train.remote(ITERATIONS))
finally:
    print("Training finished. Shutting down...")
    ray.kill(collector, no_restart=True)
    ray.shutdown()
    print("Shutdown complete.")
