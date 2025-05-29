import ray
import unstable

# tmp
import time 

ray.init()
# set the relevant parameters
batch_size = 64


lora_config = {
    "lora_rank": 32,

}
vllm_config = {
    "model_name": "Qwen/Qwen3-4B-base",
    "temperature": 0.7,
    "max_tokens": 4096,
    "max_parallel_seq": 256,
    "max_loras": 5,
    "lora_config": lora_config
}


# initialize the tracker to keep wandb up to date and print as necessary
# tracker = unstable.Tracker()

# initialize the StepBuffer (used to hold and sample from collected traces)
step_buffer = unstable.StepBuffer.remote(
    max_buffer_size = batch_size*2,
    final_reward_transformation = None, #final_reward_transformation,
    step_reward_transformation = None, #step_reward_transformation,
    sampling_reward_transformation = None, #sampling_reward_transformation,
)


# initialize and populate the Opponent pool
model_pool = unstable.ModelPool.remote(sample_mode="fixed")
# model_pool.add_checkpoint.remote(path="", iteration="-1") # add previous checkpoint
model_pool.add_fixed.remote(name="google/gemini-2.0-flash-lite-001") # add fixed opponents
model_pool.add_fixed.remote(name="google/gemini-2.0-flash-001") # add fixed opponents


collector = unstable.Collector.options(name="Collector").remote(
    num_actors=3,
    vllm_config=vllm_config,
    step_buffer=step_buffer,
    tracker=None, #tracker,
    model_pool=model_pool,
    training_envs=[("SimpleTak-v0-train", 2, "qwen3-zs")]
)


# trainer = ...
try:
    collector.collect.remote(num_workers=128)
    # trainer.train()

    time.sleep(300)

    collector.stop.remote()
finally:
    ray.kill(collector, no_restart=True)
    ray.shutdown()





# # initialize Collector
# collector = unstable.Collector(
#     num_actors = 2, step_buffer = step_buffer,
#     training_envs = [("SimpleTak-v0-train", 2, "zero_sum_game_play")],
#     evaluation_envs = [("SimpleTak-v0-train", 2, "zero_sum_game_play"), ("AIME24-v0", 1, "standard_reasoning")],
#     opponent_strategy = "self-play", vllm_dict = {},
#     num_collection_workers = 384, num_evaluation_workers = 32
# )


# # initialize the model and the learner
# learning_algorithm = unstable.algorithms.Reinforce(
#     model=model, tokenizer=tokenizer, device=device
# )


# # # initialize the Trainer
# # learner = unstable.Learner(
# #     num_learners = 1, 
# #     step_buffer = step_buffer,
# #     model_name = "Qwen/Qwen3-4B-base"
# #     algorithm = learning_algorithm,
# #     iterations = 1000
# # )


# # start both loops (one for Collector, one for Trainer)
# collector.collect() # will keep collecting until trainer is done with the designated number of iterations
# # trainer.train() # will keep training when data is available until "iterations" many update steps



# # TODO split up the rewards when saving the training data 



# # TODO buffer needs to build the output dir and communicate it to everybody else
# # TODO buffer needs to hold the logging object (such that both the collector and learner can log stuff via it)

