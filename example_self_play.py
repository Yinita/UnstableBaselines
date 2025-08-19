import time, ray, unstable
import unstable.reward_transformations as retra
import os
import logging
import sys

# Import the patch for OpenAI agent support
from patch_collector_for_openai import patch_collector_for_openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('self_play_debug.log')
    ]
)
logger = logging.getLogger("self_play")

# Configuration
COLLECTION_WORKERS = 32  # Reduced from 64 for testing
EVALUATION_WORKERS = 4   # Reduced from 8 for testing
ITERATIONS = 50
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

# ===== SELF-PLAY CONFIGURATION =====
# Choose one of the available models for self-play
# Options: "gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-chat", "gpt-5-mini", "gpt-5-nano"
SELF_PLAY_MODEL = "gpt-4o-mini"
SELF_PLAY_OPPONENT_NAME = f"openai-{SELF_PLAY_MODEL}"
logger.info(f"Using self-play model: {SELF_PLAY_MODEL} with opponent name: {SELF_PLAY_OPPONENT_NAME}")

# OpenAI configuration - simplified as we're using the new approach
openai_config = {
    "model_name": SELF_PLAY_MODEL,
    "verbose": True,
}
logger.info(f"OpenAI config: {openai_config}")

# Memory optimization: Reduced LoRA rank
lora_config = {
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj"]
}

# Memory optimization: Reduced parallel sequences
vllm_config = {
    "model_name": MODEL_NAME, 
    "temperature": 0.6, 
    "max_tokens": 4096,
    "max_parallel_seq": 32,
    "max_loras": 4,
    "lora_config": lora_config,
    "max_model_len": 16000
}

# Apply the patch with debug output
logger.info("Applying OpenAI agent patch...")
try:
    patch_collector_for_openai(openai_config)
    logger.info("Patch applied successfully!")
except Exception as e:
    logger.error(f"Failed to apply patch: {e}", exc_info=True)
    raise

# Ray init with memory management configuration
logger.info("Initializing Ray...")
try:
    ray.init(
        namespace="unstable",
        num_gpus=4,  # Specify 4 A100 GPUs
        _memory=2**33,  # 8GB memory limit
        object_store_memory=2**33,  # 8GB object store memory
        object_spilling_directory="/tmp/ray_spill"
    )
    logger.info("Ray initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize Ray: {e}", exc_info=True)
    raise

# Environment sampler configuration
logger.info("Configuring environment sampler...")
try:
    env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
        train_env_specs=[
            unstable.TrainEnvSpec(env_id="Codenames-v0", num_players=4, num_actors=4, prompt_template="qwen3-no-reasoning"),
            unstable.TrainEnvSpec(env_id="SecretMafia-v0", num_players=6, num_actors=6, prompt_template="qwen3-no-reasoning"),
            unstable.TrainEnvSpec(env_id="ThreePlayerIPD-v0", num_players=3, num_actors=3, prompt_template="qwen3-no-reasoning"),
            unstable.TrainEnvSpec(env_id="ColonelBlotto-v0", num_players=2, num_actors=2, prompt_template="qwen3-no-reasoning"),
        ],
        eval_env_specs=[
            # All opponents use the same model in self-play
            unstable.EvalEnvSpec(env_id="Codenames-v0", num_players=4, prompt_template="qwen3-no-reasoning", fixed_opponent=SELF_PLAY_OPPONENT_NAME),
            unstable.EvalEnvSpec(env_id="SecretMafia-v0", num_players=6, prompt_template="qwen3-no-reasoning", fixed_opponent=SELF_PLAY_OPPONENT_NAME),
            unstable.EvalEnvSpec(env_id="ThreePlayerIPD-v0", num_players=3, prompt_template="qwen3-no-reasoning", fixed_opponent=SELF_PLAY_OPPONENT_NAME),
            unstable.EvalEnvSpec(env_id="ColonelBlotto-v0", num_players=2, prompt_template="qwen3-no-reasoning", fixed_opponent=SELF_PLAY_OPPONENT_NAME),
        ],
    )
    logger.info(f"Environment sampler configured with environments: {env_sampler.env_list()}")
except Exception as e:
    logger.error(f"Failed to configure environment sampler: {e}", exc_info=True)
    raise

# Tracker
logger.info("Initializing tracker...")
try:
    tracker = unstable.Tracker.options(name="Tracker").remote(
        run_name=f"SelfPlay-{SELF_PLAY_MODEL}-{MODEL_NAME.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", 
        wandb_project="UnstableBaselines"
    )
    logger.info("Tracker initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize tracker: {e}", exc_info=True)
    raise

# Initialize model registry
logger.info("Initializing model registry...")
try:
    model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
    ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
    
    # Add our OpenAI agent as a fixed opponent
    logger.info(f"Adding fixed opponent: {SELF_PLAY_OPPONENT_NAME}")
    ray.get(model_registry.add_fixed.remote(name=SELF_PLAY_OPPONENT_NAME))
    logger.info("Model registry initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize model registry: {e}", exc_info=True)
    raise

# Initialize model sampler
logger.info("Initializing model sampler...")
try:
    model_sampler = unstable.samplers.model_samplers.BaseModelSampler(model_registry=model_registry)
    logger.info("Model sampler initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize model sampler: {e}", exc_info=True)
    raise

# Build game scheduler
logger.info("Building game scheduler...")
try:
    game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(
        model_sampler=model_sampler, 
        env_sampler=env_sampler, 
        logging_dir=ray.get(tracker.get_log_dir.remote())
    )
    logger.info("Game scheduler built successfully!")
except Exception as e:
    logger.error(f"Failed to build game scheduler: {e}", exc_info=True)
    raise

# Data Buffer
logger.info("Initializing data buffer...")
try:
    step_buffer = unstable.EpisodeBuffer.options(name="Buffer").remote(
        max_buffer_size=256, 
        tracker=tracker,
        final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
        step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
        sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
    )
    logger.info("Data buffer initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize data buffer: {e}", exc_info=True)
    raise

# Initialize the collector
logger.info("Initializing collector...")
try:
    collector = unstable.Collector.options(name="Collector").remote(
        vllm_config=vllm_config, 
        tracker=tracker, 
        buffer=step_buffer, 
        game_scheduler=game_scheduler,
    )
    logger.info("Collector initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize collector: {e}", exc_info=True)
    raise

# Initialize the learner
logger.info("Initializing learner...")
try:
    learner = unstable.A2CLearner.options(
        num_gpus=4,
        name="Learner",
        memory=2**33,
        object_store_memory=2**33
    ).remote(
        model_name=MODEL_NAME,
        lora_cfg=lora_config,
        batch_size=128,
        mini_batch_size=1,
        learning_rate=1e-5,
        grad_clip=0.2,
        buffer=step_buffer,
        tracker=tracker,
        model_registry=model_registry,
        activation_checkpointing=True,
        gradient_checkpointing=True,
        use_trainer_cache=False,
    )
    logger.info("Learner initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize learner: {e}", exc_info=True)
    raise

# Initialize algorithm
logger.info("Initializing algorithm...")
try:
    ray.get(learner.initialize_algorithm.remote(
        infer_mini_batch_size=8,
        critic_learning_rate=5e-5,
        normalize_adv=True,
        max_train_len=10000,
        max_generation_len=4096,
    ))
    logger.info("Algorithm initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize algorithm: {e}", exc_info=True)
    raise

# Run training
logger.info(f"Starting training with {COLLECTION_WORKERS} collection workers and {EVALUATION_WORKERS} evaluation workers...")
try:
    collector.collect.remote(num_train_workers=COLLECTION_WORKERS, num_eval_workers=EVALUATION_WORKERS)
    logger.info("Collection started successfully!")
    
    logger.info(f"Starting training for {ITERATIONS} iterations...")
    ray.get(learner.train.remote(ITERATIONS))
    logger.info("Training completed successfully!")
except Exception as e:
    logger.error(f"Error during training: {e}", exc_info=True)
    raise
finally:
    logger.info("Cleaning up resources...")
    ray.kill(collector, no_restart=True)
    ray.shutdown()
    logger.info("Cleanup completed!")

logger.info("Self-play training script completed successfully!")
