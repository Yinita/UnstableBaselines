import time, ray, unstable
import unstable.reward_transformations as retra
import sys
import os

# Add the envs directory to Python path so we can import the custom environment
sys.path.append('/home/aiscuser/mindgames')

# Import the patch for OpenAI agent support
from patch_collector_for_openai import patch_collector_for_openai

# Import and register the ThreePlayerIPD environment
from envs.ThreePlayerIPD import *  # This will register the environment

# Training configuration
COLLECTION_WORKERS = 200
EVALUATION_WORKERS = 16
ITERATIONS = 200
MODEL_NAME = "Qwen/Qwen3-8B"
BATCH_SIZE = 384
MINI_BATCH_SIZE = 1
BUFFER_SIZE = 384*2
LR = 1e-5
GRAD_CLIP = 0.2
MAX_TRAIN_SEQ_LEN = 30000
MAX_GENERATION_LENGTH = 30000

# GSPO specific parameters
GROUP_SIZE = 4  # Number of responses per query for group-based advantage estimation
CLIP_RATIO = 0.2  # Sequence-level clipping ratio
NORMALIZE_LENGTH = True  # Whether to apply length normalization to importance ratios

# OpenAI Agent configuration
OPENAI_MODEL_NAME = "gpt-4o"  # You can change this to any OpenAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Custom opponent identifier for OpenAI agent
OPENAI_OPPONENT_NAME = "openai-gpt4o"

lora_config = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
vllm_config = {
    "model_name": MODEL_NAME, "temperature": 0.6, "max_tokens": MAX_GENERATION_LENGTH,
    "max_parallel_seq": 128, "max_loras": 8, "lora_config": lora_config,
    "max_model_len": 30000
}

# Apply the patch to support OpenAI agents BEFORE initializing Ray
print("Applying OpenAI agent patch to UnstableBaselines collector...")
openai_config = {
    "model_name": OPENAI_MODEL_NAME,
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
    "verbose": True
}
patch_collector_for_openai(openai_config)

# Ray init
ray.init(namespace="unstable")  

print(f"Initializing GSPO training for Three Player Iterated Prisoner's Dilemma...")
print(f"  - Training Model: {MODEL_NAME}")
print(f"  - Opponent Model: {OPENAI_MODEL_NAME} (OpenAI)")
print(f"  - Game: Three Player Iterated Prisoner's Dilemma")
print(f"  - Players per game: 3")
print(f"  - OpenAI Base URL: {OPENAI_BASE_URL}")

# Initialize environment sampler with ThreePlayerIPD
env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
    train_env_specs=[
        # Training games: 3 players, all training models
        unstable.TrainEnvSpec(
            env_id="ThreePlayerIPD-v0-train", 
            num_players=3, 
            num_actors=3,  # All 3 players are training models
            prompt_template="qwen3-zs"
        ),
    ],
    eval_env_specs=[
        # Evaluation games: 2 training models + 1 OpenAI opponent
        unstable.EvalEnvSpec(
            env_id="ThreePlayerIPD-v0", 
            num_players=3, 
            prompt_template="qwen3-zs", 
            fixed_opponent=OPENAI_OPPONENT_NAME
        ),
        # Additional evaluation: standard 2-player setup for comparison
        # unstable.EvalEnvSpec(
        #     env_id="KuhnPoker-v0-train", 
        #     num_players=2, 
        #     prompt_template="qwen3-zs", 
        #     fixed_opponent=OPENAI_OPPONENT_NAME
        # ),
    ]
)

# Tracker
tracker = unstable.Tracker.options(name="Tracker").remote(
    run_name=f"GSPO-ThreePlayerIPD-{MODEL_NAME.split('/')[-1]}-{int(time.time())}", 
    wandb_project="UnstableBaselines-ThreePlayerIPD"
) 

# Initialize model registry
model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))

# Add our OpenAI agent as a fixed opponent
ray.get(model_registry.add_fixed.remote(name=OPENAI_OPPONENT_NAME))

# Initialize model sampler
model_sampler = unstable.samplers.model_samplers.BaseModelSampler(model_registry=model_registry) 

# Build game scheduler
game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(
    model_sampler=model_sampler, 
    env_sampler=env_sampler, 
    logging_dir=ray.get(tracker.get_log_dir.remote())
)

# Data Buffer - GSPO uses EpisodeBuffer like A2C
step_buffer = unstable.EpisodeBuffer.options(name="Buffer").remote(
    max_buffer_size=BUFFER_SIZE, tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

# Initialize the collector
collector = unstable.Collector.options(name="Collector").remote(
    vllm_config=vllm_config, tracker=tracker, buffer=step_buffer, game_scheduler=game_scheduler,
)

# Initialize the GSPO learner
learner = unstable.GSPOLearner.options(num_gpus=1, name="Learner").remote(
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

# Initialize GSPO algorithm with specific parameters
ray.get(learner.initialize_algorithm.remote(
    infer_mini_batch_size=32,
    group_size=GROUP_SIZE,
    clip_ratio=CLIP_RATIO,
    normalize_length=NORMALIZE_LENGTH,
    normalize_adv=True,
    max_train_len=MAX_TRAIN_SEQ_LEN,
    max_generation_len=MAX_GENERATION_LENGTH
))

print(f"\nStarting GSPO training for Three Player Iterated Prisoner's Dilemma:")
print(f"  - Training Model: {MODEL_NAME}")
print(f"  - Opponent: {OPENAI_OPPONENT_NAME} ({OPENAI_MODEL_NAME})")
print(f"  - Game Environment: ThreePlayerIPD-v0-train")
print(f"  - Players per training game: 3 (all training models)")
print(f"  - Players per eval game: 3 (2 training + 1 OpenAI opponent)")
print(f"  - Group size: {GROUP_SIZE}")
print(f"  - Clip ratio: {CLIP_RATIO}")
print(f"  - Length normalization: {NORMALIZE_LENGTH}")
print(f"  - Collection workers: {COLLECTION_WORKERS}")
print(f"  - Evaluation workers: {EVALUATION_WORKERS}")
print(f"  - Training iterations: {ITERATIONS}")
print(f"  - Max sequence length: {MAX_TRAIN_SEQ_LEN}")
print(f"  - Max generation length: {MAX_GENERATION_LENGTH}")

print("\nGame Rules:")
print("  - 3-player Iterated Prisoner's Dilemma")
print("  - 10 rounds per game (training), 5 rounds per game (evaluation)")
print("  - Each round: 2-3 communication turns + decision phase")
print("  - Players choose cooperate/defect for each opponent")
print("  - Payoff matrix: CC=3, DD=1, CD=0, DC=5")
print("  - Goal: Maximize total score across all rounds")

print("\nThe collector has been patched to use your OpenAI agent instead of OpenRouter!")
print("Your OpenAI agent will participate as the third player in evaluation games.")

try:
    collector.collect.remote(num_train_workers=COLLECTION_WORKERS, num_eval_workers=EVALUATION_WORKERS)
    ray.get(learner.train.remote(ITERATIONS))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()
