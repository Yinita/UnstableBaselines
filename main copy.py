import time, ray, unstable
import unstable.reward_transformations as retra

NUM_LEARNERS = 2
NUM_ACTORS = 1
COLLECTION_WORKERS = 128
EVALUATION_WORKERS = 0
ITERATIONS = 128
MODEL_NAME = "Qwen/Qwen3-4B-base"
BATCH_SIZE = 4
BUFFER_SIZE = 32*2
GRAD_ACCUM = 4
LR = 1e-5
GRAD_CLIP = 0.2

lora_config = {
    "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj","down_proj"]
}
vllm_config = {
    "model_name": MODEL_NAME, "temperature": 0.6, "max_tokens": 4096,
    "max_parallel_seq": 384, "max_loras": 50, "lora_config": lora_config
}

TRAINING_ENVS = [("SimpleTak-v0-train", 2, "qwen3-zs")]
EVALUATION_ENVS= [("SimpleTak-v0-train", 2, "qwen3-zs")]

WANDB_RUN_NAME = f"Run--{MODEL_NAME.split('/')[-1]}-{int(time.time())}"

# Ray init 
ray.init(log_to_driver=True, logging_level="DEBUG")

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
learners = [
    unstable.Learner.options(name=f"Learner-{r}", num_gpus=1).remote(
        rank=r, world_size=NUM_LEARNERS, model_name=MODEL_NAME, step_buffer=step_buffer, model_pool=model_pool, algorithm=algorithm, 
        batch_size=BATCH_SIZE, gradient_accum=GRAD_ACCUM, lr=LR, grad_clip=GRAD_CLIP, delay_mult=1.5, lora_cfg=lora_config, tracker=tracker
    )
    for r in range(NUM_LEARNERS)
]

try:
    collector.collect.remote(num_workers=COLLECTION_WORKERS, num_eval_workers=EVALUATION_WORKERS) # start collection
    ray.get([L.ready.remote() for L in learners]) # ensure everybody is ready
    ray.get([L.synced.remote() for L in learners])
    ray.get([L.train.remote(ITERATIONS) for L in learners]) # start learning
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()
