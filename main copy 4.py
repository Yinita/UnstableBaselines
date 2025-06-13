import time, ray, unstable
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import unstable.reward_transformations as retra

NUM_LEARNERS = 1
NUM_ACTORS = 3
COLLECTION_WORKERS = 128
EVALUATION_WORKERS = 0
ITERATIONS = 128
MODEL_NAME = "Qwen/Qwen3-4B-base"
BATCH_SIZE = 2
BUFFER_SIZE = 2*2
GRAD_ACCUM = 2
LR = 1e-5
GRAD_CLIP = 0.2

lora_config = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
vllm_config = {
    "model_name": MODEL_NAME, "temperature": 0.6, "max_tokens": 4096,
    "max_parallel_seq": 32, "max_loras": 5, "lora_config": lora_config
}

TRAINING_ENVS = [("SimpleTak-v0-train", 2, "qwen3-zs")]
EVALUATION_ENVS= [("SimpleTak-v0-train", 2, "qwen3-zs")]

WANDB_RUN_NAME = f"Run--{MODEL_NAME.split('/')[-1]}-{int(time.time())}"

# Ray init 
# ray.init(log_to_driver=True, logging_level="DEBUG")
ray.init(log_to_driver=True)

# Tracker
tracker = unstable.Tracker.options(name="Tracker").remote(run_name=WANDB_RUN_NAME, wandb_project="UnstableBaselines")

# Data Buffer
step_buffer = unstable.StepBuffer.remote(
    max_buffer_size=BUFFER_SIZE, tracker=tracker,
    final_reward_transformation=retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()]),
    step_reward_transformation=retra.ComposeStepRewardTransforms([retra.RewardForThinkTags(1.5), retra.PenaltyForInvalidMove(1.0, -1.0)]),
    sampling_reward_transformation=retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(True)]),
)

# Model Pool
model_pool = unstable.ModelPool.remote(tracker=tracker, sample_mode="mirror", max_active_lora=5)
ray.get(model_pool.add_checkpoint.remote(path=None, iteration="-1"))
ray.get(model_pool.add_fixed.remote(name="google/gemini-2.0-flash-lite-001"))

# Collector
collector = unstable.Collector.options(name="Collector").remote(
    num_actors=NUM_ACTORS, tracker=tracker, vllm_config=vllm_config, step_buffer=step_buffer, model_pool=model_pool,
    training_envs=TRAINING_ENVS, evaluation_envs=EVALUATION_ENVS, evaluation_opponent="google/gemini-2.0-flash-lite-001"
)

# Algorithm and Learner
algorithm = unstable.algorithms.Reinforce()


deepspeed_config = {
    "train_batch_size": BATCH_SIZE * GRAD_ACCUM,
    "gradient_accumulation_steps": GRAD_ACCUM,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": LR,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    },
    "fp16": {"enabled": False},
    "bf16": {"enabled": True},  # if using BF16
}


training_config = {
    "deepspeed_config": deepspeed_config,
    "step_buffer": step_buffer,
    "tracker": tracker,
    "model_pool": model_pool,
    "iterations": 64,
    "lora_cfg": lora_config,
    "model_name": vllm_config["model_name"],
    "batch_size": BATCH_SIZE
}
trainer = TorchTrainer(
    unstable.learners.ds_learning_function.train_func,
    train_loop_config=training_config,
    scaling_config=ScalingConfig(num_workers=2, use_gpu=True), # resources_per_worker={"GPU": 2} for tensor parallel
)

try:
    collector.collect.remote(num_workers=COLLECTION_WORKERS, num_eval_workers=EVALUATION_WORKERS)
    trainer.fit()
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()
