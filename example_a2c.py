import time, ray, unstable
import unstable.reward_transformations as retra
import os

# Import the patch for OpenAI agent support
from patch_collector_for_openai import patch_collector_for_openai
# always uses 1 learner and the remainder of the GPUS as actors
COLLECTION_WORKERS = 64  # Reduced from 200 to lower memory pressure
EVALUATION_WORKERS = 8   # Reduced from 16
ITERATIONS = 200
MODEL_NAME = "Qwen/Qwen3-8B-Base"
OPENAI_OPPONENT_NAME = "openai-gpt4o"

# Memory optimization: Reduced batch sizes
BATCH_SIZE = 128         # Reduced from 384
MINI_BATCH_SIZE = 1      # Keep at 1 for best results
INFER_MINI_BATCH_SIZE = 8  # New parameter for inference batching
BUFFER_SIZE = 256        # Reduced buffer size

# Training parameters
LR = 1e-5
GRAD_CLIP = 0.2
MAX_TRAIN_SEQ_LEN = 10000  # Keep this as is for context length
MAX_GENERATION_LENGTH = 4096

# Memory optimization: Enable 8-bit and 4-bit quantization
USE_8BIT_QUANT = True    # Use 8-bit quantization for policy model
USE_4BIT_QUANT = True    # Use 4-bit quantization for critic model

# OpenAI configuration
OPENAI_MODEL_NAME = "gpt-4o"  # You can change this to any OpenAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
openai_config = {
    "model_name": OPENAI_MODEL_NAME,
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
    "verbose": True,
}

# Memory optimization: Reduced LoRA rank
lora_config = {
    "lora_rank": 16,       # Reduced from 32
    "lora_alpha": 16,      # Reduced from 32
    "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj"] # ,"o_proj","gate_proj", "up_proj","down_proj"
}

# Memory optimization: Reduced parallel sequences
vllm_config = {
    "model_name": MODEL_NAME, 
    "temperature": 0.6, 
    "max_tokens": MAX_GENERATION_LENGTH,
    "max_parallel_seq": 32,  # Reduced from 128
    "max_loras": 4,          # Reduced from 8
    "lora_config": lora_config,
    "max_model_len": 16000
}

patch_collector_for_openai(openai_config)

# Ray init
# Ray init with memory management configuration
ray.init(
    namespace="unstable",
    _memory=2**33,  # 8GB memory limit
    object_store_memory=2**33,  # 8GB object store memory
    _system_config={
        "object_spilling_config": "{\"type\": \"filesystem\"}",
        "max_direct_call_object_size": 2**30,  # 1GB
    }
)
env_sampler = unstable.samplers.env_samplers.UniformRandomEnvSampler(
    train_env_specs=[
        unstable.TrainEnvSpec(env_id="Codenames-v0", num_players=4, num_actors=4, prompt_template="qwen3-no-reasoning"),
        unstable.TrainEnvSpec(env_id="SecretMafia-v0", num_players=6, num_actors=6, prompt_template="qwen3-no-reasoning"),
        unstable.TrainEnvSpec(env_id="ThreePlayerIPD-v0", num_players=3, num_actors=3, prompt_template="qwen3-no-reasoning"),
        unstable.TrainEnvSpec(env_id="ColonelBlotto-v0", num_players=2, num_actors=2, prompt_template="qwen3-no-reasoning"),
    ],
    eval_env_specs=[
        unstable.EvalEnvSpec(env_id="Codenames-v0", num_players=4, prompt_template="qwen3-no-reasoning", fixed_opponent=OPENAI_OPPONENT_NAME),
        unstable.EvalEnvSpec(env_id="SecretMafia-v0", num_players=6, prompt_template="qwen3-no-reasoning", fixed_opponent=OPENAI_OPPONENT_NAME),
        unstable.EvalEnvSpec(env_id="ThreePlayerIPD-v0", num_players=3, prompt_template="qwen3-no-reasoning", fixed_opponent=OPENAI_OPPONENT_NAME),
        unstable.EvalEnvSpec(env_id="ColonelBlotto-v0", num_players=2, prompt_template="qwen3-no-reasoning", fixed_opponent=OPENAI_OPPONENT_NAME),
    ],
)


# Tracker
tracker = unstable.Tracker.options(name="Tracker").remote(
    run_name=f"Test-{MODEL_NAME.split('/')[-1]}-{env_sampler.env_list()}-{int(time.time())}", 
    wandb_project="UnstableBaselines"
) 

# initialize model registry
model_registry = unstable.ModelRegistry.options(name="ModelRegistry").remote(tracker=tracker)
ray.get(model_registry.add_checkpoint.remote(uid="base", path=None, iteration=0))
# Add our OpenAI agent as a fixed opponent
ray.get(model_registry.add_fixed.remote(name=OPENAI_OPPONENT_NAME))

# initialize model sampler
model_sampler = unstable.samplers.model_samplers.BaseModelSampler(model_registry=model_registry) 

# build game scheduler
game_scheduler = unstable.GameScheduler.options(name="GameScheduler").remote(model_sampler=model_sampler, env_sampler=env_sampler, logging_dir=ray.get(tracker.get_log_dir.remote()))

# Data Buffer
step_buffer = unstable.EpisodeBuffer.options(name="Buffer").remote(
    max_buffer_size=BUFFER_SIZE, tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForFormat(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

# initialize the collector
collector = unstable.Collector.options(name="Collector").remote(
    vllm_config=vllm_config, tracker=tracker, buffer=step_buffer, game_scheduler=game_scheduler,
)

# Memory optimization: Configure Ray to use object spilling to disk
ray.init(namespace="unstable", _memory=2**33, object_store_memory=2**33, _system_config={"object_spilling_config": "{\"type\": \"filesystem\"}"})

# initialize the learner with memory optimizations
learner = unstable.A2CLearner.options(
    num_gpus=4,  # Use all 4 GPUs for model parallelism
    name="Learner",
    memory=2**33,  # 8GB memory limit
    object_store_memory=2**33  # 8GB object store memory
).remote(
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
    use_trainer_cache=False,
    # Memory optimization: Additional parameters
    use_8bit_quant=USE_8BIT_QUANT,
    use_4bit_quant=USE_4BIT_QUANT,
    offload_to_cpu=True  # Offload parameters to CPU when not in use
)

# Initialize with smaller inference batch size
ray.get(learner.initialize_algorithm.remote(
    infer_mini_batch_size=INFER_MINI_BATCH_SIZE,  # Reduced from 32
    critic_learning_rate=5e-5,
    normalize_adv=True,
    max_train_len=MAX_TRAIN_SEQ_LEN,
    max_generation_len=MAX_GENERATION_LENGTH,
    share_backbone=True  # Share backbone between policy and critic to save memory
))


try:
    collector.collect.remote(num_train_workers=COLLECTION_WORKERS, num_eval_workers=EVALUATION_WORKERS)
    ray.get(learner.train.remote(ITERATIONS))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()