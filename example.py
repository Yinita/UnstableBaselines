import ray, time
import unstable
import unstable.reward_transformations as retra

# set the relevant parameters
NUM_ACTORS = 2
NUM_LEARNERS = 1
COLLECTION_WORKERS = 64 #512 
EVALUTION_WORKERS = 0
ITERATIONS = 500

MODEL_NAME = "Qwen/Qwen3-1.7B-base"

BATCH_SIZE = 16
BUFFER_SIZE = 64*2
GRADIENT_ACCUMULATION_STEPS = 16

LR = 1e-5
GRAD_CLIP = 0.2


lora_config = {"lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
vllm_config = {"model_name": MODEL_NAME, "temperature": 0.6, "max_tokens": 4096, "max_parallel_seq": 384, "max_loras": 5, "lora_config": lora_config}

TRAINING_ENVS = [
    # ("LiarsDice-v0-train", 2, "qwen3-zs"), 
    ("SimpleTak-v0-train", 2, "qwen3-zs"), 
    # ("Nim-v0-train", 2, "qwen3-zs"), 
    # ("KuhnPoker-v0-train", 2, "qwen3-zs"), 
    # ("SimpleNegotiation-v0-train", 2, "qwen3-zs")
]
EVALUATION_ENVS = [
    # ("LiarsDice-v0-train", 2, "qwen3-zs"), 
    # ("SimpleTak-v0-train", 2, "qwen3-zs"), 
    # ("Nim-v0-train", 2, "qwen3-zs"), 
    # ("KuhnPoker-v0-train", 2, "qwen3-zs"), 
    # ("SimpleNegotiation-v0-train", 2, "qwen3-zs")
]

WANDB_RUN_NAME = f"Batch-0-Experiment-0--{MODEL_NAME.split('/')[-1]}-[{','.join([t[0] for t in TRAINING_ENVS])}]-{int(time.time())}"


ray.init()

# initialize the tracker to keep wandb up to date and print as necessary
tracker = unstable.Tracker.options(name="Tracker").remote(run_name=WANDB_RUN_NAME, wandb_project="UnstableBaselines")


# build the reward transformations to be used
final_reward_transformation = retra.ComposeFinalRewardTransforms([retra.RoleAdvantageByEnvFormatter()])
step_reward_transformation = retra.ComposeStepRewardTransforms([retra.RewardForThinkTags(reward=1.5), retra.PenaltyForInvalidMove(reward= 1.0, penalty= -1.0)])
sampling_reward_transformation = retra.ComposeSamplingRewardTransforms([retra.NormalizeRewardsByEnv(z_score=True)])

# initialize the StepBuffer (used to hold and sample from collected traces)
step_buffer = unstable.StepBuffer.remote(
    max_buffer_size=BUFFER_SIZE, tracker=tracker,
    final_reward_transformation=final_reward_transformation, #final_reward_transformation,
    step_reward_transformation=step_reward_transformation, #step_reward_transformation,
    sampling_reward_transformation=sampling_reward_transformation, #sampling_reward_transformation,
)

# initialize and populate the Opponent pool
model_pool = unstable.ModelPool.remote(tracker=tracker, sample_mode="mirror", max_active_lora=5)
ray.get(model_pool.add_checkpoint.remote(path=None, iteration="-1")) # add base checkpoint
ray.get(model_pool.add_fixed.remote(name="google/gemini-2.0-flash-lite-001")) # add fixed opponents

# build collector
collector = unstable.Collector.options(name="Collector").remote(
    num_actors=NUM_ACTORS, tracker=tracker, vllm_config=vllm_config, step_buffer=step_buffer, model_pool=model_pool, 
    training_envs=TRAINING_ENVS, evaluation_envs=EVALUATION_ENVS, evaluation_opponent="google/gemini-2.0-flash-lite-001"
)

# build learner
algorithm = unstable.algorithms.Reinforce() # if you use something else, this is the place to pass stuff
learner = unstable.Learner.options(num_gpus=NUM_LEARNERS, name="Learner").remote(
    num_learners=NUM_LEARNERS, tracker=tracker, model_name=MODEL_NAME, step_buffer=step_buffer, model_pool=model_pool, algorithm=algorithm, 
    batch_size=BATCH_SIZE, gradient_accum_steps=GRADIENT_ACCUMULATION_STEPS, learning_rate=LR, grad_clip=GRAD_CLIP, batch_delay_buffer=1.5, lora_cfg=lora_config,
)

try:
    collector.collect.remote(num_workers=COLLECTION_WORKERS, num_eval_workers=EVALUTION_WORKERS)
    ray.get(learner.train.remote(ITERATIONS))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()
