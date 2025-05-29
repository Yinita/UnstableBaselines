import ray, unstable
import time 

# set the relevant parameters
NUM_ACTORS = 2
NUM_LEARNERS = 1

MODEL_NAME = "Qwen/Qwen3-4B-base"

BATCH_SIZE = 16
BUFFER_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = 16

LR = 5e-5
GRAD_CLIP = 0.5


lora_config = {
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}

vllm_config = {
    "model_name": MODEL_NAME, 
    "temperature": 0.7,
    "max_tokens": 4096,
    "max_parallel_seq": 256,
    "max_loras": 5,
    "lora_config": lora_config
}
TRAINING_ENVS = [("SimpleTak-v0-train", 2, "qwen3-zs")]

ray.init()


# initialize the tracker to keep wandb up to date and print as necessary
tracker = unstable.WandBTracker.options(name="Tracker").remote(wandb_run_name=f"{MODEL_NAME.split('/')[-1]}-[{','.join([t[0] for t in TRAINING_ENVS])}]-{int(time.time())}")

# initialize the StepBuffer (used to hold and sample from collected traces)
step_buffer = unstable.StepBuffer.remote(
    max_buffer_size = BUFFER_SIZE,
    final_reward_transformation = None, #final_reward_transformation,
    step_reward_transformation = None, #step_reward_transformation,
    sampling_reward_transformation = None, #sampling_reward_transformation,
)


# initialize and populate the Opponent pool
model_pool = unstable.ModelPool.remote(sample_mode="adaptive-trueskill")
ray.get(model_pool.add_checkpoint.remote(path=None, iteration="-1")) # add base checkpoint
# model_pool.add_checkpoint.remote(path="", iteration="-1") # add previous checkpoint
ray.get(model_pool.add_fixed.remote(name="google/gemini-2.0-flash-lite-001")) # add fixed opponents
ray.get(model_pool.add_fixed.remote(name="google/gemini-2.0-flash-001")) # add fixed opponents

# build collector
collector = unstable.Collector.options(name="Collector").remote(
    num_actors=NUM_ACTORS,
    tracker=tracker,
    vllm_config=vllm_config,
    step_buffer=step_buffer,
    model_pool=model_pool,
    training_envs=TRAINING_ENVS
)


# build learner
learner = unstable.Learner.options(num_gpus=NUM_LEARNERS, name="Learner").remote(
    num_learners=NUM_LEARNERS,
    tracker=tracker,
    model_name=MODEL_NAME,
    step_buffer=step_buffer,
    model_pool=model_pool,
    algorithm=unstable.algorithms.Reinforce(), # if you use something else, this is the place to pass stuff
    batch_size=BATCH_SIZE,
    gradient_accum_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LR,
    grad_clip=GRAD_CLIP,
    batch_delay_buffer=1.5,
    lora_cfg=lora_config,
)



try:
    collector.collect.remote(num_workers=8)
    ray.get(learner.train.remote(800))
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()



# # TODO split up the rewards when saving the training data 
# # TODO buffer needs to build the output dir and communicate it to everybody else
# # TODO buffer needs to hold the logging object (such that both the collector and learner can log stuff via it)

# # TODO log trueskills
# # TODO log opponent frequency 
# # TODO add grad_norm back into tracking